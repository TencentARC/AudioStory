# --------------------------------------------------------
# AudioStory: Generating Long-Form Narrative Audio with Large Language Models（arxiv.org/pdf/2508.20088）
# ARC Lab, Tencent PCG
# Github: https://github.com/TencentARC/AudioStory
# Licensed under The Apache License [see LICENSE for details]
# Based on SEED-X code base
# --------------------------------------------------------


"""
AudioStory model.

- Computes LM loss, reconstruction loss (optional), and DiT loss
- Fuses T5 and audio embeddings via Multi-Head Attention
- Supports text-to-audio inference with fused embeddings
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import load_zero3_checkpoint

BOA_TOKEN = "<aud>"
EOA_TOKEN = "</aud>"
AUD_TOKEN = "<aud_{:05d}>"

GEN_BOS_TOKEN = "<|gen|>"
GEN_EOS_TOKEN = "<|/gen|>"

THINK_BOS_TOKEN = "<|think|>"
THINK_EOS_TOKEN = "<|/think|>"


def cosine_loss(rec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance loss: mean(1 - cos_sim).

    Uses a numerically-stable normalization with eps to avoid NaNs for zero vectors.
    """
    eps = 1e-8
    target = F.normalize(target, p=2, dim=-1, eps=eps)
    rec = F.normalize(rec, p=2, dim=-1, eps=eps)
    return (1.0 - (target * rec).sum(dim=-1)).mean()


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class AudioStory_unified(nn.Module):
    """
    SeedX + DiT with T5/audio-token attention fusion.

    Args:
        seedx: Upstream multimodal model that returns hidden states and LM loss.
        tangoflux: Audio generator/DiT module (callable for training & inference).
        rec_loss_type: One of {"mse", "kl", "cos_sim", None}. If None, reconstruction loss is skipped.
        model_dims: (Unused here; kept for API compatibility).
        lm_loss_scale: Scale for LM loss.
        dit_loss_scale: Scale for DiT loss.
        rec_loss_scale: Scale for reconstruction loss.
    """

    def __init__(
        self,
        seedx: nn.Module,
        tangoflux: nn.Module,
        rec_loss_type: Optional[str] = "mse",
        model_dims: int = 5120,
        lm_loss_scale: float = 1.0,
        dit_loss_scale: float = 1.0,
        rec_loss_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.seedx = seedx
        self.tangoflux = tangoflux
        self.model_dims = model_dims
        self.rec_loss_type = rec_loss_type
        print("*********** Use rec_loss_type:", self.rec_loss_type, "***********")
        self.lm_loss_scale = lm_loss_scale
        self.dit_loss_scale = dit_loss_scale
        self.rec_loss_scale = rec_loss_scale

        self.num_attn_heads = 8
        self.audio_embedding_dim = 1024
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.audio_embedding_dim, num_heads=self.num_attn_heads
        )

    def build_audio_projector_layernorm(
        self, hidden_size: int, target_hidden_size: int
    ) -> nn.Sequential:
        """
        Simple LN -> MLP projector. Not used by default; kept for experimentation.
        """
        print("Build LayerNorm-based audio projector.")
        return nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, target_hidden_size),
            nn.GELU(),
            nn.Linear(target_hidden_size, target_hidden_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        image_embeds: Optional[torch.Tensor],
        audio_embeds: Optional[torch.Tensor],
        beats_embeds: Optional[torch.Tensor],
        caption_embeds: torch.Tensor,
        embeds_gen_mask: torch.Tensor,
        embeds_cmp_mask: torch.Tensor,
        ids_t5_gen_mask: torch.Tensor,
        ids_aud_gen_mask: torch.Tensor,
        ids_cmp_mask: torch.Tensor,
        audio_latent: torch.Tensor,
        duration: torch.Tensor | float,
        freeze_llm: bool,  # unused here but kept for API compatibility
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing LM loss, (optional) reconstruction loss, and DiT loss.

        Returns:
            {
              "total_loss": scalar,
              "lm_loss": scalar,
              "audio_rec_loss": scalar,
              "audio_dit_loss": scalar
            }
        """
        (
            recon_t5_embeds,
            recon_audio_embeds,
            lm_loss,
        ) = self.seedx.get_last_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_embeds=image_embeds,
            audio_embeds=audio_embeds,
            beats_embeds=beats_embeds,
            embeds_gen_mask=embeds_gen_mask,
            embeds_cmp_mask=embeds_cmp_mask,
            ids_t5_gen_mask=ids_t5_gen_mask,
            ids_aud_gen_mask=ids_aud_gen_mask,
            ids_cmp_mask=ids_cmp_mask,
            caption_embeds=caption_embeds,
        )


        t5_transposed = recon_t5_embeds.permute(1, 0, 2)  
        aud_transposed = recon_audio_embeds.permute(1, 0, 2) 
        attn_output = self.multihead_attention(
            t5_transposed, aud_transposed, aud_transposed
        )[0].permute(1, 0, 2)                                  
        t5_fused_embeds = (recon_t5_embeds + attn_output) / self.seedx.t5_feature_scale

        audio_rec_loss = lm_loss.new_zeros(())
        if self.rec_loss_type is not None:
            cap = caption_embeds.detach().clone()
            cap = cap.view(cap.size(0), -1, self.audio_embedding_dim) * self.seedx.t5_feature_scale

            if self.rec_loss_type == "mse":
                audio_rec_loss = F.mse_loss(recon_t5_embeds, cap)

        audio_dit_loss, _, _, _ = self.tangoflux(audio_latent, t5_fused_embeds, duration=duration)

        total_loss = (
            self.lm_loss_scale * lm_loss
            + self.rec_loss_scale * audio_rec_loss
            + self.dit_loss_scale * audio_dit_loss
        )

        return {
            "total_loss": total_loss,
            "lm_loss": lm_loss,
            "audio_rec_loss": audio_rec_loss,
            "audio_dit_loss": audio_dit_loss,
        }

    @torch.no_grad()
    def inference_audiostory_tta(
        self,
        prompt: str,
        tokenizer, 
        input_ids: torch.Tensor,
        num_t5_gen_tokens: int,
        num_aud_gen_tokens: int,
        max_new_tokens: int = 1600,
        steps: int = 25,
        duration: float = 10.0,
        guidance_scale: float = 4.5,
    ) -> torch.Tensor:
        """
        Text-to-audio inference.

        Returns:
            Latent tensor from TangoFlux (to be decoded by the VAE).
        """
        recon_t5_embeds, recon_audio_embeds = self.seedx.generate_T5_audtoken_attn_multi_audio(
            tokenizer=tokenizer,
            input_ids=input_ids,
            num_t5_gen_tokens=num_t5_gen_tokens,
            num_aud_gen_tokens=num_aud_gen_tokens,
            max_new_tokens=max_new_tokens,
        )

        t5_transposed = recon_t5_embeds.permute(1, 0, 2)     
        aud_transposed = recon_audio_embeds.permute(1, 0, 2) 
        attn_output = self.multihead_attention(
            t5_transposed, aud_transposed, aud_transposed
        )[0].permute(1, 0, 2)
        t5_fused_embeds = (recon_t5_embeds + attn_output) / self.seedx.t5_feature_scale

        latents = self.tangoflux.inference_flow_full_tokens(
            prompt,
            t5_fused_embeds,
            duration=duration,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )
        return latents


    @classmethod
    def from_pretrained(
        cls,
        seedx: nn.Module,
        tangoflux: nn.Module,
        pretrained_model_path: Optional[str] = None,
        **kwargs,
    ) -> "SEED_X_DIT_T5_Audiotoken_attn_multi_audio_coscale":
        """
        Construct model and optionally load a ZeRO-3 style checkpoint.

        If environment variable DEBUG_FLAG=True, loading is skipped.
        """
        model = cls(seedx=seedx, tangoflux=tangoflux, **kwargs)

        if os.environ.get("DEBUG_FLAG", "False") == "True":
            return model

        if pretrained_model_path:
            ckpt = torch.load(pretrained_model_path, map_location="cpu")
            load_zero3_checkpoint(model, ckpt)

        return model