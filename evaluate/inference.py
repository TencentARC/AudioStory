#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Long-form audio generation inference code.

- Load configs via Hydra/OmegaConf
- Generate multiple audio clips per prompt
- Decode with VAE, resample, and save
- Smoothly concatenate clips with crossfade into one file
- Provide Mel filterbank utilities (Kaldi fbank) for downstream use
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import librosa
import torch
import torchaudio
from omegaconf import OmegaConf

# Project imports
from src.models.detokenizer.modeling_flux import (  # noqa: E402
    Flux_T5 as FluxDetokenizerT5,
)

# Models / utils
from diffusers import AutoencoderOobleck  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

# Silence noisy warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Constants / global config
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
MEL_BINS = 128
TARGET_LEN = 1024
NORM_MEAN = -4.2677393
NORM_STD = 4.5689974

# Prompt bank for variety
GEN_PROMPT_ALL = [
    "Create a long-form audio composition that explores ",
    "I'd like an extended audio track with rich details of ",
    "Craft an immersive soundscape that slowly unfolds ",
    "Please produce a comprehensive recording capturing every aspect of ",
    "Generate a detailed, drawn-out audio that brings to life ",
    "Let's design a textured audio piece highlighting the nuances of ",
    "Kindly fashion a lengthy audio journey through ",
    "Devise a long-form audio narrative of ",
    "Develop a gradually developing audio with meticulous attention to ",
    "Imagine and produce a vivid audio experience of ",
    "Construct a detailed soundscape that tells the story of ",
    "Please generate a sustained audio track filled with ",
    "Produce an in-depth, multi-layered audio that portrays ",
    "Design a descriptive audio capturing the essence of ",
    "Create a nuanced, extended audio that conveys ",
    "I'd like to commission a long-form audio depicting ",
    "Fashion a richly detailed audio recording of ",
    "Propose a detailed, immersive audio that showcases ",
    "Generate an elaborate audio piece focusing on ",
    "Develop a comprehensive audio that fully represents ",
]

# ---------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------
def smooth_concatenate(
    audio_tensors: List[torch.Tensor],
    sample_rate: int,
    transition_sec: float = 3.0,
) -> torch.Tensor:
    """
    Smoothly concatenate multiple audio tensors using a linear crossfade.

    Args:
        audio_tensors: List of audio tensors shaped [channels, samples] or [samples].
        sample_rate: Sampling rate of inputs.
        transition_sec: Crossfade duration in seconds.

    Returns:
        Tensor [channels, total_samples] with smooth transitions.
        Returns an empty float32 tensor if the list is empty.
    """
    if not audio_tensors:
        return torch.empty(0, dtype=torch.float32)

    # Normalize shapes and dtypes
    proc: List[torch.Tensor] = []
    for a in audio_tensors:
        if a.dim() == 1:
            a = a.unsqueeze(0)  # [1, T]
        elif a.dim() != 2:
            raise ValueError("Each audio tensor must be 1D or 2D (C x T).")
        proc.append(a.to(dtype=torch.float32))

    transition_samples = max(0, int(transition_sec * sample_rate))
    result = proc[0]

    if transition_samples == 0:
        return torch.cat(proc, dim=1)

    for nxt in proc[1:]:
        if result.shape[1] < transition_samples or nxt.shape[1] < transition_samples:
            result = torch.cat([result, nxt], dim=1)
            continue

        overlap_a = result[:, -transition_samples:]
        overlap_b = nxt[:, :transition_samples]

        device = result.device
        fade_out = torch.linspace(1.0, 0.0, transition_samples, device=device).unsqueeze(0)
        fade_in = torch.linspace(0.0, 1.0, transition_samples, device=device).unsqueeze(0)

        merged = overlap_a * fade_out + overlap_b * fade_in
        result = torch.cat([result[:, :-transition_samples], merged, nxt[:, transition_samples:]], dim=1)

    return result


def process_and_prepare_concat(
    audio_tensors: List[torch.Tensor],
    original_sample_rate: int = 44100,
    transition_sec: float = 3.0,
) -> torch.Tensor:
    """
    Build a single crossfaded track from a list of clips.

    Args:
        audio_tensors: List of per-clip tensors (C x T) or (T,).
        original_sample_rate: Sampling rate of the inputs.
        transition_sec: Crossfade length in seconds.

    Returns:
        Float32 tensor of the concatenated audio.
    """
    return smooth_concatenate(
        audio_tensors, sample_rate=original_sample_rate, transition_sec=transition_sec
    )


# ---------------------------------------------------------------------
# FBank utilities
# ---------------------------------------------------------------------
def wav2fbank(filename: str) -> torch.Tensor:
    """
    Compute Kaldi-style mel filterbank features for a WAV file.

    Returns:
        Tensor of shape [TARGET_LEN, MEL_BINS], padded/truncated as needed.
    """
    waveform, sr = torchaudio.load(filename)  # [C, T], float32
    waveform = waveform - waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=MEL_BINS,
        dither=0.0,
        frame_shift=10,  # ms
    )

    # Pad/crop to fixed length
    n_frames = fbank.shape[0]
    pad = TARGET_LEN - n_frames
    if pad > 0:
        padder = torch.nn.ZeroPad2d((0, 0, 0, pad))
        fbank = padder(fbank)
    elif pad < 0:
        fbank = fbank[:TARGET_LEN, :]

    return fbank


def norm_fbank(fbank: torch.Tensor) -> torch.Tensor:
    """Normalize fbank features (AudioSet-style normalization)."""
    return (fbank - NORM_MEAN) / (NORM_STD * 2)


def prepare_one_fbank(wav_path: str, cuda_enabled: bool = True) -> Dict[str, torch.Tensor]:
    """
    Prepare a normalized fbank sample dictionary for a single WAV file.

    Args:
        wav_path: Path to the WAV file.
        cuda_enabled: Unused; kept for API compatibility.

    Returns:
        {"fbank": <normalized fbank tensor>}
    """
    fbank = norm_fbank(wav2fbank(wav_path))
    return {"fbank": fbank}


# ---------------------------------------------------------------------
# Caption parsing
# ---------------------------------------------------------------------
def extract_content_and_duration(text: str) -> Tuple[str, Optional[float]]:
    """
    Extract content outside of <timestamp>...</timestamp> and parse duration inside.

    Args:
        text: Caption that may contain a <timestamp>seconds</timestamp> tag.

    Returns:
        (content_without_tag, duration_in_seconds_or_None)
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
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio generation runner")

    # Runtime
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    # Model configs
    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "ckpt/audiostory_3b"
        ),
    )
    parser.add_argument(
        "--llm_cfg_path",
        type=str,
        default="configs/audiostory_llm_qwen25_3b_lora.yaml",
    )
    parser.add_argument(
        "--tangoflux_model",
        type=str,
        default="ckpt/Flux_detokenizer",
    )

    # Generation
    parser.add_argument("--audio_type", default="demo")
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--max_new_tokens", type=float, default=1600)
    parser.add_argument("--target_sr", type=int, default=32000, help="Output sample rate")
    parser.add_argument("--crossfade_sec", type=float, default=3, help="Crossfade seconds")

    # IO / prompt
    parser.add_argument("--save_folder_name", type=str, default="audiostory_multi_audio_duration")
    parser.add_argument("--generated_caption", type=str, default="")
    parser.add_argument("--total_duration", type=float, default=50)

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except Exception:
        pass

    # Output directory
    model_name = Path(args.model_path).parts[-2] + "/" + Path(args.model_path).parts[-1]
    base_out = Path("evaluate") / args.save_folder_name
    inference_save_dir = base_out / f"generated_audio_{args.audio_type}_{args.guidance}"
    inference_save_dir.mkdir(parents=True, exist_ok=True)

    print("************************************************")
    print("Inference models:", args.model_path)
    print("Save dir:", str(inference_save_dir))
    print("************************************************\n")

    # --------------------- Initialize models ---------------------
    model_config = OmegaConf.load(f"{args.model_path}/config.yaml")
    tokenizer_cfg = model_config.tokenizer
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    if model_config.llm_model is not None:
        llm_cfg = model_config.llm_model
    else:
        print("model_config.llm_model is None! Falling back to provided LLM cfg path.")
        llm_cfg = args.llm_cfg_path
        print("LLM config:", args.llm_cfg_path)

    llm = hydra.utils.instantiate(llm_cfg, torch_dtype=DTYPE)
    print("Init LLM done.")

    agent_model_cfg = model_config.agent_model
    agent_model_cfg.pretrained_model_path = None
    agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)
    agent_model.eval().to(DEVICE, dtype=DTYPE)
    print("Init agent model done.")

    # Detokenizer / TangoFlux
    with open(f"{args.tangoflux_model}/config.json", "r") as f:
        flux_config = json.load(f)

    flux_model = FluxDetokenizerT5(config=flux_config)
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

    # VAE (Oobleck)
    vae = AutoencoderOobleck()
    weights = load_file(f"{args.tangoflux_model}/model.safetensors")
    vae.load_state_dict(weights, strict=True)
    vae.eval().to(DEVICE, dtype=DTYPE)
    print("Init VAE model done.")

    # --------------------- Build instruction ---------------------
    lead = random.choice(GEN_PROMPT_ALL)
    whole_caption = (lead + args.generated_caption).strip()
    instruction_with_duration = f"{whole_caption} The total duration is {args.total_duration} seconds."
    prompt = SYSTEM_MESSAGE + INSTRUCTION_SFT_PROMPT.format_map({"instruction": instruction_with_duration}) + ASSISTANT_ANSWER
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    print("\n****************************************")
    print("Instruction:", instruction_with_duration)
    print("Output dir:", str(inference_save_dir))
    print("max_new_tokens:", args.max_new_tokens)
    print("num_t5_out_tokens:", model_config.train_dataset.datapipes[0].num_t5_out_tokens)

    # --------------------- Generate tokens ----------------------
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

    print("Generated reasoning:", generated_reasoning)

    if output_t5_tokens is None or caption_list is None or len(caption_list) == 0:
        print("Warning: No tokens or captions produced. Exiting.")
        return

    # Fuse T5 & audio tokens via attention
    t5_transposed = output_t5_tokens.permute(1, 0, 2)
    audio_transposed = output_aud_tokens.permute(1, 0, 2)
    attn_output = seedx_dit_model.multihead_attention(
        t5_transposed, audio_transposed, audio_transposed
    )[0].permute(1, 0, 2)
    t5_fused_embeds = (output_t5_tokens + attn_output) / seedx_dit_model.seedx.t5_feature_scale

    multi_audio_num = len(output_t5_tokens)
    print(f"Generated {multi_audio_num} caption segments.")
    audio_tensors: List[torch.Tensor] = []

    # --------------------- Decode each clip ---------------------
    for i in range(multi_audio_num):
        caption_i = caption_list[i]
        content_i, duration_i = extract_content_and_duration(caption_i)
        if i >= multi_audio_num - 1:
                duration = float(duration_i)
        else:
            duration = float(duration_i) + args.crossfade_sec

        print(f"Caption_{i}: {content_i}  |  duration≈{duration:.2f}s")

        t5_fused_embed = t5_fused_embeds[i].unsqueeze(0)

        with torch.no_grad():
            output_latents = seedx_dit_model.tangoflux.inference_flow_full_tokens(
                content_i,
                t5_fused_embed,
                duration=duration,
                num_inference_steps=50,
                guidance_scale=args.guidance,
            )

            # Decode with VAE at its native sampling rate
            wave = vae.decode(output_latents.transpose(2, 1)).sample.cpu()[0]  # [C, T]
            waveform_end = int(duration * vae.config.sampling_rate)
            wave = wave[:, :waveform_end]  # exact trimming

        # Resample to target SR for saving (e.g., 32 kHz)
        wave_np = wave.numpy().astype("float32")  # [C, T]
        resampled = librosa.resample(
            wave_np, orig_sr=vae.config.sampling_rate, target_sr=args.target_sr, axis=-1
        )
        wave_resampled = torch.tensor(resampled, dtype=torch.float32)
        audio_tensors.append(wave_resampled)

    if not audio_tensors:
        print("Warning: No audio decoded; nothing to save.")
        return

    # --------------------- Concatenate & save -------------------
    concat_name = "generated_audio.wav"
    concatenated_audio = process_and_prepare_concat(
        audio_tensors, original_sample_rate=args.target_sr, transition_sec=args.crossfade_sec
    )
    torchaudio.save(str(inference_save_dir / concat_name), concatenated_audio, sample_rate=args.target_sr)
    print(f"Saved: {inference_save_dir / concat_name}")


if __name__ == "__main__":
    main()