#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-clip audio generation and smooth concatenation.

- Load configs via Hydra/OmegaConf
- Generate multiple audio clips per prompt
- Decode with VAE, resample, and (optionally) save per-clip audio
- Crossfade-concatenate clips and save the merged track
- Provide Kaldi fbank helpers (fixed length, normalized)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import librosa
import torch
import torchaudio
from omegaconf import OmegaConf

from src.models.detokenizer.tangoflux_t5_tokens import (  # noqa: E402
    TangoFlux_T5 as TangoFluxDetokenizerT5,
)

from diffusers import AutoencoderOobleck  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

# Silence noisy warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Globals / constants
# ---------------------------------------------------------------------

BOI_TOKEN = "<t5>"
EOI_TOKEN = "</t5>"
AUD_TOKEN = "<t5_{:05d}>"

QWEN_BOS_INDEX = 151644
QWEN_EOS_INDEX = 151645

DEVICE = "cuda:0"
DTYPE = torch.float16

SYSTEM_MESSAGE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
INSTRUCTION_SFT_PROMPT = "<|im_start|>user\n{instruction}<|im_end|>\n"
ASSISTANT_ANSWER = "<|im_start|>assistant\n<|think|>"

# FBank defaults
MELBINS = 128
TARGET_LEN = 1024
NORM_MEAN = -4.2677393
NORM_STD = 4.5689974


# ---------------------------------------------------------------------
# Small utility: robustly extract content and duration from captions
# ---------------------------------------------------------------------
def extract_content_and_duration(text: str) -> Tuple[str, float | None]:
    """
    Parse `...<timestamp>SEC</timestamp>...` inside a caption.

    Returns:
        (content_without_tags, duration_in_seconds_or_None)
    """
    start_tag, end_tag = "<timestamp>", "</timestamp>"
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return text.strip(), None

    content = (text[:start_idx] + text[end_idx + len(end_tag) :]).strip()
    dur_str = text[start_idx + len(start_tag) : end_idx].strip()
    try:
        duration = float(dur_str) if dur_str else None
    except ValueError:
        duration = None
    return content, duration


# ---------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------
def smooth_concatenate(
    audio_tensors: List[torch.Tensor],
    sample_rate: int,
    transition_sec: float = 3.0,
) -> torch.Tensor:
    """
    Smoothly concatenate audio clips with a linear crossfade.

    Args:
        audio_tensors: list of [C, T] (or [T]) tensors.
        sample_rate: sampling rate of inputs.
        transition_sec: crossfade length in seconds.

    Returns:
        [C, T_total] float32 tensor. Empty tensor if input list is empty.
    """
    if not audio_tensors:
        return torch.empty(0, dtype=torch.float32)

    # Normalize shapes and dtype
    proc: List[torch.Tensor] = []
    for a in audio_tensors:
        if a.dim() == 1:
            a = a.unsqueeze(0)  # -> [1, T]
        elif a.dim() != 2:
            raise ValueError("Each audio tensor must be 1D or 2D (C x T).")
        proc.append(a.to(dtype=torch.float32))

    transition_samples = max(0, int(transition_sec * sample_rate))
    result = proc[0]

    if transition_samples == 0:
        return torch.cat(proc, dim=1)

    for nxt in proc[1:]:
        # Fallback to hard concat if either segment is shorter than the crossfade
        if result.shape[1] < transition_samples or nxt.shape[1] < transition_samples:
            result = torch.cat([result, nxt], dim=1)
            continue

        overlap_a = result[:, -transition_samples:]
        overlap_b = nxt[:, :transition_samples]

        device = result.device
        fade_out = torch.linspace(1.0, 0.0, transition_samples, device=device).unsqueeze(0)  # [1, T]
        fade_in = torch.linspace(0.0, 1.0, transition_samples, device=device).unsqueeze(0)   # [1, T]

        merged = overlap_a * fade_out + overlap_b * fade_in
        result = torch.cat([result[:, :-transition_samples], merged, nxt[:, transition_samples:]], dim=1)

    return result


def process_and_save_audio(
    audio_tensors: List[torch.Tensor],
    output_path: str,
    original_sample_rate: int = 44100,
    transition_sec: float = 3.0,
) -> torch.Tensor:
    """
    Prepare a crossfaded concatenated tensor. (The function name kept for compatibility.)

    Note:
        `output_path` is kept to preserve the original signature; we only ensure
        the directory exists here and return the concatenated tensor. Saving is
        handled by the caller.

    Returns:
        Concatenated audio tensor (float32).
    """
    # Ensure output directory exists (mirrors original behavior)
    outdir = os.path.dirname(output_path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    concatenated_audio = smooth_concatenate(
        audio_tensors, sample_rate=original_sample_rate, transition_sec=transition_sec
    )
    return concatenated_audio


# ---------------------------------------------------------------------
# FBank utilities
# ---------------------------------------------------------------------
def wav2fbank(filename: str) -> torch.Tensor:
    """
    Compute Kaldi-style mel filterbank features for a WAV file.

    Returns:
        [TARGET_LEN, MELBINS] tensor padded/cropped to fixed length.
    """
    waveform, sr = torchaudio.load(filename)  # [C, T], float32
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=MELBINS,
        dither=0.0,
        frame_shift=10,  # ms
    )

    n_frames = fbank.shape[0]
    p = TARGET_LEN - n_frames
    if p > 0:
        padder = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = padder(fbank)
    elif p < 0:
        fbank = fbank[:TARGET_LEN, :]

    return fbank


def norm_fbank(fbank: torch.Tensor) -> torch.Tensor:
    """Normalize fbank features (AudioSet-style normalization)."""
    return (fbank - NORM_MEAN) / (NORM_STD * 2)


def prepare_one_fbank(wav_path: str, cuda_enabled: bool = True) -> Dict[str, torch.Tensor]:
    """
    Prepare a normalized fbank sample dictionary for a single WAV file.

    Args:
        wav_path: path to .wav file.
        cuda_enabled: unused; kept for API compatibility.

    Returns:
        {"fbank": <normalized fbank tensor>}
    """
    f = norm_fbank(wav2fbank(wav_path))
    return {"fbank": f}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="clotho")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    # Qwen
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "audioseed_ckpt/seed_omni_t5_multi_audio_aud_attn_new/"
            "seed_omni_qwen_3b_t5_multi_audio_unav_scale10_1e4_"
            "loss0105_bz8_genpretrain_withinst_t5_aud_attn_cotrain_with_mhattn_"
            "weight_detokenizer_full_open_1opt_coscale_8token_duration_begin0/checkpoint-12000"
        ),
    )
    parser.add_argument(
        "--llm_cfg_path",
        type=str,
        default="configs/ablation_studies/t2a_whisper/llm_audioseed_qwen25_3b_lora.yaml",
    )

    # Generation
    parser.add_argument(
        "--tangoflux_model",
        type=str,
        default="a_TangoFlux_trained_ckpt_40075/new_detokenizer/tangoflux_detokenizer_30s_700k_1e5/step_165000",
    )
    parser.add_argument("--audio_type", default="audio_gen_multi_audio_UnAV_gemini")
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--max_new_tokens", type=float, default=1600)
    parser.add_argument("--crossfade_sec", type=float, default=3, help="Crossfade seconds")

    parser.add_argument(
        "--testset_json",
        default="datasets_audio_json/eval/eval_multi_audio_unav_testset_duration_dounew.json",
    )
    parser.add_argument("--save_folder_name", type=str, default="audiostory_multi_audio_duration")
    parser.add_argument("--feature_scale", type=float, default=None)
    args = parser.parse_args()

    # ---------------- Misc init ----------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = Path(args.model_path).parts[-2] + "/" + Path(args.model_path).parts[-1]
    base_out = Path("audioseed_tangoflux_generate_FAD_results") / args.save_folder_name / model_name
    save_dir = base_out / f"{args.audio_type}_{args.guidance}"
    save_dir_concat = base_out / f"{args.audio_type}_{args.guidance}_concat"
    json_path = base_out / f"{model_name.replace('/', '_')}_{args.audio_type}_{args.guidance}_result.json"

    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_concat.mkdir(parents=True, exist_ok=True)

    print("******************************************")
    print("Inference models:", args.model_path)
    print("Inference feature scale:", args.feature_scale)
    print("\n")

    # ---------------- Models ----------------
    model_config = OmegaConf.load(f"{args.model_path}/config.yaml")
    tokenizer_cfg = model_config.tokenizer
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    if model_config.llm_model is not None:
        llm_cfg = model_config.llm_model
    else:
        print("model_config.llm_model is None!")
        llm_cfg = args.llm_cfg_path
        print("LLM config:", args.llm_cfg_path)

    llm = hydra.utils.instantiate(llm_cfg, torch_dtype=DTYPE)
    print("Init llm done.")

    agent_model_cfg = model_config.agent_model
    agent_model_cfg.pretrained_model_path = None
    agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)
    agent_model.eval().to(DEVICE, dtype=DTYPE)
    print("Init agent model done.")

    # TangoFlux detokenizer
    model_path = args.tangoflux_model
    with open(f"{model_path}/config.json", "r") as f:
        flux_config = json.load(f)

    flux_model = TangoFluxDetokenizerT5(config=flux_config)
    flux_model.eval().to(DEVICE, dtype=DTYPE)

    # SeedX-DiT
    seedx_dit_model_cfg = model_config.seedx_dit_model_cfg
    if not seedx_dit_model_cfg._target_.endswith("from_pretrained"):
        seedx_dit_model_cfg._target_ = seedx_dit_model_cfg._target_ + ".from_pretrained"
    seedx_dit_model_cfg.pretrained_model_path = f"{args.model_path}/pytorch_model.bin"
    seedx_dit_model = hydra.utils.instantiate(
        seedx_dit_model_cfg, seedx=agent_model, tangoflux=flux_model
    )
    seedx_dit_model.eval().to(DEVICE, dtype=DTYPE)
    print("Init seedx_dit model done.")

    # VAE
    vae = AutoencoderOobleck()
    weights = load_file(f"{args.tangoflux_model}/model.safetensors")
    vae.load_state_dict(weights, strict=True)
    vae.eval().to(DEVICE, dtype=DTYPE)
    print("Init vae model done.")

    # ---------------- Generate ----------------
    with open(args.testset_json, "r") as f:
        data = json.load(f)

    counter = 0
    for element in data:
        # Read fields
        captions = element["whole_caption"]
        ids = counter
        audio_clips = element["audio_clips"]

        # Build LLM prompt
        prompt = SYSTEM_MESSAGE + INSTRUCTION_SFT_PROMPT.format_map({"instruction": captions}) + ASSISTANT_ANSWER
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        print("\n")
        print("****************************************")
        print("Caption:", captions)
        print("save dir:", str(save_dir))
        print("max_new_tokens:", args.max_new_tokens)
        print("num_t5_out_tokens:", model_config.train_dataset.datapipes[0].num_t5_out_tokens)

        # Generate tokens
        with torch.no_grad():
            (
                output_t5_tokens,
                output_aud_tokens,
                caption_list,
                generated_reasoning,
            ) = seedx_dit_model.seedx.generate_T5_audtoken_attn_multi_audio(
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                num_t5_gen_tokens=model_config.train_dataset.datapipes[0].num_t5_out_tokens,
                num_aud_gen_tokens=model_config.train_dataset.datapipes[0].num_aud_out_tokens,
            )

        if output_t5_tokens is not None:
            # Fuse T5 & audio tokens
            t5_transposed = output_t5_tokens.permute(1, 0, 2)
            audio_transposed = output_aud_tokens.permute(1, 0, 2)
            attn_output = seedx_dit_model.multihead_attention(
                t5_transposed, audio_transposed, audio_transposed
            )[0].permute(1, 0, 2)
            t5_fused_embeds = (output_t5_tokens + attn_output) / seedx_dit_model.seedx.t5_feature_scale

            multi_audio_num = len(output_t5_tokens)
            audio_tensors: List[torch.Tensor] = []

            for i in range(multi_audio_num):
                # Caption & duration
                caption_i = caption_list[i]
                print(f"Caption_{i}: {caption_i}")

                # Extract content and per-clip duration (with small headroom)
                content_i, duration_i = extract_content_and_duration(caption_i)
                if i >= multi_audio_num - 1:
                    duration = float(duration_i)
                else:
                    duration = float(duration_i) + args.crossfade_sec

                t5_fused_embed = t5_fused_embeds[i].unsqueeze(0)

                with torch.no_grad():
                    output_latents = seedx_dit_model.tangoflux.inference_flow_full_tokens(
                        content_i,
                        t5_fused_embed,
                        duration=duration,
                        num_inference_steps=50,
                        guidance_scale=args.guidance,
                    )

                    wave = vae.decode(output_latents.transpose(2, 1)).sample.cpu()[0]  # [C, T]
                    waveform_end = int(duration * vae.config.sampling_rate)
                    wave = wave[:, :waveform_end]

                # Resample to 32 kHz for saving (axis=-1 to act along time)
                wave_np = wave.numpy().astype("float32")  # [C, T]
                wave_resampled_np = librosa.resample(
                    wave_np, orig_sr=vae.config.sampling_rate, target_sr=32000, axis=-1
                )
                wave_resampled = torch.tensor(wave_resampled_np, dtype=torch.float32)
                audio_tensors.append(wave_resampled)

                # If you also want to save per-clip files, uncomment below:
                # clip_name = f"id{ids}_segment{i+1}.wav"
                # torchaudio.save(str(save_dir / clip_name), wave_resampled, sample_rate=32000)

            # Concatenate and save merged result
            concat_name = f"{ids}_concated_clips.wav"
            concatenated_audio = process_and_save_audio(
                audio_tensors, str(save_dir_concat), original_sample_rate=32000, transition_sec=args.crossfade_sec
            )
            torchaudio.save(str(save_dir_concat / concat_name), concatenated_audio, sample_rate=32000)

            counter += 1
            element["generated_reasoning"] = generated_reasoning
            element["generated_caption"] = [{"caption": c} for c in caption_list]

        if counter % 10 == 0:
            print("\n\n******************************************")
            print(f"Completed {counter} audio files.")
            print("******************************************\n\n\n")

        # Persist running JSON results
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)