# --------------------------------------------------------
# AudioStory: Generating Long-Form Narrative Audio with Large Language Models（arxiv.org/pdf/2508.20088）
# ARC Lab, Tencent PCG
# Github: https://github.com/TencentARC/AudioStory
# Licensed under The Apache License [see LICENSE for details]
# Based on SEED-X code base
# --------------------------------------------------------

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LogitsProcessorList
from .generation import AutoAudioTokenGenerationProcessor, AutoT5TokenGenerationProcessor
from .utils import load_zero3_checkpoint

import numpy as np
import math
import re

from torch.nn.init import trunc_normal_


BOA_TOKEN = '<aud>'
EOA_TOKEN = '</aud>'
AUD_TOKEN = '<aud_{:05d}>'

BOT_TOKEN = '<t5>'
EOT_TOKEN = '</t5>'
T5_TOKEN = '<t5_{:05d}>'

GEN_BOS_TOKEN="<|gen|>"
GEN_EOS_TOKEN="<|/gen|>"

THINK_BOS_TOKEN="<|think|>"
THINK_EOS_TOKEN="<|/think|>"



def cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


def get_2d_sincos_pos_embed(embed_dim, h_size, w_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(h_size, dtype=np.float32)
    grid_w = np.arange(w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, h_size, w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1) 
    out = np.einsum('m,d->md', pos, omega)  

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


def get_abs_pos(abs_pos, tgt_size):
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos






class AudioStory_llm(nn.Module):

    def __init__(self, llm, input_resampler, output_resampler, whisper_resampler_llava, t5_feature_scale=10.0, audio_feature_scale=1.0, target_audio_type='audiomae', model_dims=5120, lm_loss_scale=1.0, t5_rec_loss_scale=1.0, audio_rec_loss_scale=1.0, add_patch_pos=False, vit_down=False, mse=False) -> None:
        super().__init__()
        self.llm = llm
        self.input_resampler = input_resampler
        self.output_resampler = output_resampler
        self.whisper_resampler_llava = whisper_resampler_llava


        self.lm_loss_scale = lm_loss_scale
        self.t5_rec_loss_scale = t5_rec_loss_scale
        self.audio_rec_loss_scale = audio_rec_loss_scale
        self.add_patch_pos = add_patch_pos

        self.t5_feature_scale = t5_feature_scale
        self.audio_feature_scale = audio_feature_scale

        print('***************************')
        print('t5_feature_scale is: ', self.t5_feature_scale)
        print('audio_feature_scale is: ', self.audio_feature_scale)
        print('lm_loss_scale is: ', self.lm_loss_scale)
        print('t5_rec_loss_scale is: ', self.t5_rec_loss_scale)
        print('audio_rec_loss_scale is: ', self.audio_rec_loss_scale)
        print('***************************')
        self.vit_down = vit_down
        if self.vit_down:
            self.pool_size = 4  
            self.stride = 4  
        
        self.mse = mse
        if self.mse:
            self.mse_loss = torch.nn.MSELoss() 

        self.add_patch_pos = add_patch_pos
        if self.add_patch_pos:
            patch_dim = self.input_resampler.embed_dim
            self.patch_pos_embed = nn.Parameter((patch_dim**-0.5) * torch.randn(4, patch_dim))


        self.model_dims = model_dims
        # self.tgt_audio_type = target_audio_type
        self.tgt_audio_dim = 1024
        


        self.projector = self.build_audio_projector('mlp2x_gelu', self.model_dims, self.tgt_audio_dim)
        self.audio_projector = self.build_audio_projector('mlp2x_gelu', self.model_dims, self.tgt_audio_dim)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def build_audio_projector(self, projector_type, hidden_size, target_hidden_size):

        if projector_type == 'linear':
            return nn.Linear(hidden_size, target_hidden_size)

        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(hidden_size, target_hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(target_hidden_size, target_hidden_size))
            return nn.Sequential(*modules)

        if projector_type == 'identity':
            return nn.Identity()

        raise ValueError(f'Unknown projector type: {projector_type}')


    def forward(self, input_ids, attention_mask, labels, image_embeds, audio_embeds, beats_embeds, caption_embeds, embeds_gen_mask, embeds_cmp_mask, ids_t5_gen_mask, ids_aud_gen_mask, 
                ids_cmp_mask, patch_positions=None):


        input_embeds = self.llm.get_input_embeddings()(input_ids)

        bz, sq, dim = input_embeds.shape

        output_lm = self.llm(attention_mask=attention_mask,
                            inputs_embeds=input_embeds,
                            labels=labels,
                            output_hidden_states=True,
                            return_dict=True)
        lm_loss = output_lm['loss']

        last_hidden_state = output_lm.hidden_states[-1]
        

        target_embeds = caption_embeds.clone().detach()

        if self.feature_scale is not None:
            target_embeds = target_embeds * self.feature_scale

        num_auds_for_rec = target_embeds.shape[0]
        target_embeds = target_embeds.view(num_auds_for_rec, -1, self.tgt_audio_dim)
        
        output_t5_embeds = last_hidden_state[ids_t5_gen_mask].view(num_auds_for_rec, -1, dim)

    
        recon_t5_embeds = self.projector(output_t5_embeds)

        if self.mse:
            audio_rec_loss = F.mse_loss(recon_t5_embeds, target_embeds.detach())
        else:
            audio_rec_loss = cosine_loss(recon_t5_embeds, target_embeds.detach())
            

        # ******************************************************************************* #

        total_loss = self.lm_loss_scale * lm_loss + self.rec_loss_scale * audio_rec_loss

        return {'total_loss': total_loss, 'lm_loss': lm_loss, 'audio_rec_loss': audio_rec_loss}



    def get_last_hidden_states(self, input_ids, attention_mask, labels, image_embeds, audio_embeds, beats_embeds, embeds_gen_mask, embeds_cmp_mask, 
                                ids_t5_gen_mask, ids_aud_gen_mask, ids_cmp_mask, caption_embeds=None, patch_positions=None):


        input_embeds = self.llm.get_input_embeddings()(input_ids)

        bz, sq, dim = input_embeds.shape

        if audio_embeds is not None:
            if audio_embeds is not None:
                audio_embeds_cmp = audio_embeds[embeds_cmp_mask]  # num_imgs_in_batch x nq_in x dim_in, 4 x 1500, 4096

        if audio_embeds is not None and audio_embeds_cmp.shape[0] > 0:
            audio_embeds_lm = self.whisper_resampler_llava(audio_embeds_cmp)  # num_imgs_in_batch x nq x dim, 4 x 64 x 4096
            has_audio_cmp = True

        else:
            audio_embeds_cmp_fake = None
            audio_embeds_lm = None
            has_audio_cmp = False


        has_audio_input = audio_embeds is not None and embeds_cmp_mask.sum().item() > 0
        has_audio_output = caption_embeds is not None and embeds_gen_mask.sum().item() > 0


        if has_audio_input:
            input_embeds[ids_cmp_mask] = audio_embeds_lm.reshape(-1, dim)  # eg, 128 x 4096
            # zero_loss = 0.0

        output_lm = self.llm(attention_mask=attention_mask,
                            inputs_embeds=input_embeds,
                            labels=labels,
                            output_hidden_states=True,
                            return_dict=True)
        lm_loss = output_lm['loss']

        last_hidden_state = output_lm.hidden_states[-1] # torch.Size([2, 256, 4096]) 256固定



        if has_audio_output:
            num_text_for_rec = input_embeds.shape[0]
            output_t5_embeds = last_hidden_state[ids_t5_gen_mask].view(num_text_for_rec, -1, dim)  # torch.Size([16, 4096]) -> torch.Size([2, 8, 4096])
            output_audio_embeds = last_hidden_state[ids_aud_gen_mask].view(num_text_for_rec, -1, dim)  # torch.Size([16, 4096]) -> torch.Size([2, 8, 4096])
            recon_t5_embeds = self.projector(output_t5_embeds) / self.t5_feature_scale# torch.Size([2, 8, 4096]) -> torch.Size([2, 8, 768])
            recon_audio_embeds = self.audio_projector(output_audio_embeds) / self.t5_feature_scale# torch.Size([2, 8, 4096]) -> torch.Size([2, 8, 768])

        else:
            recon_t5_embeds = None
            recon_audio_embeds = None


        return recon_t5_embeds, recon_audio_embeds, lm_loss

        
    
    def generate_T5_audtoken_attn_multi_audio(self,
                 tokenizer,
                 prompt=None,
                 input_ids=None,
                 image_embeds=None,
                 audio_embeds=None,
                 ids_gen_mask=None,
                 logits_processor=None,
                 num_t5_gen_tokens=64,
                 num_aud_gen_tokens=8,
                 temperature=0.7,
                 num_beams=1,
                 max_new_tokens=300,
                 top_p=0.5,
                 dtype=torch.float16,
                 device='cuda:0',
                 patch_positions=None):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(
                AutoT5TokenGenerationProcessor(tokenizer=tokenizer, num_t5_gen_tokens=num_t5_gen_tokens))
            logits_processor.append(
                AutoAudioTokenGenerationProcessor(tokenizer=tokenizer, num_aud_gen_tokens=num_aud_gen_tokens))

        # assert prompt.size()[0]==1
        if prompt is not None:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(device=device)
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        bz, sq, dim = input_embeds.shape

        generation_config = {
            'temperature': temperature,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'do_sample': False
        }

        # generate_ids = self.llm.generate(input_ids=input_ids, **generation_config)
        output = self.llm.generate(input_ids=input_ids,
                                   inputs_embeds=input_embeds,
                                   output_hidden_states=True,
                                   return_dict_in_generate=True,
                                   logits_processor=logits_processor,
                                   **generation_config)

        generate_ids = output.sequences[0][input_ids.shape[1]:]
        # print('generate_ids:', generate_ids)
        print('generate_text:', tokenizer.decode(generate_ids))
        generate_id_list = generate_ids.tolist()


        # GEN_BOS_TOKEN
        gen_bos_token_id = tokenizer.encode(GEN_BOS_TOKEN, add_special_tokens=False)[0]
        aud_end_token_id = tokenizer.encode(EOA_TOKEN, add_special_tokens=False)[0]
        eot_token_id = tokenizer.encode(EOT_TOKEN, add_special_tokens=False)[0]
        bot_token_id = tokenizer.encode('<>', add_special_tokens=False)[0]
        think_eos_token_id = tokenizer.encode(THINK_EOS_TOKEN, add_special_tokens=False)[0]
        

        last_hidden_states = torch.cat([hidden_state[-1] for hidden_state in output.hidden_states],
                                       dim=1)[0, input_ids.shape[1]:, :]

        gen_bos_token_id = torch.where(generate_ids == gen_bos_token_id)[0].tolist()
        eot_token_id = torch.where(generate_ids == eot_token_id)[0].tolist()
        bot_token_id = torch.where(generate_ids == bot_token_id)[0].tolist()
        aud_end_token_id = torch.where(generate_ids == aud_end_token_id)[0].tolist()
        think_eos_token_id = torch.where(generate_ids == think_eos_token_id)[0].tolist()

        try:
            if len(think_eos_token_id) > 0:
                generated_reasoning = tokenizer.decode(generate_ids[:think_eos_token_id[0]])
            else:
                generated_reasoning = tokenizer.decode(generate_ids[:bot_token_id[0]])
        

            has_aud_output = True
            if has_aud_output:
                caption_list = []
                t5_gen_feats = []
                aud_gen_feats = []
                for i in range(len(eot_token_id)):
                    eoi_idx = eot_token_id[i]
                    gen_bos_idx = gen_bos_token_id[i]
                    aud_end_token_idx = aud_end_token_id[i]
                    caption_list.append(tokenizer.decode(generate_ids[gen_bos_idx+1:eoi_idx - num_t5_gen_tokens-1]))
                    t5_gen_feats.append(last_hidden_states[eoi_idx - num_t5_gen_tokens:eoi_idx])
                    aud_gen_feats.append(last_hidden_states[aud_end_token_idx - num_aud_gen_tokens:aud_end_token_idx])

                t5_gen_feats = torch.stack(t5_gen_feats)
                aud_gen_feats = torch.stack(aud_gen_feats)
                t5_gen_feat = self.projector(t5_gen_feats)
                aud_gen_feat = self.audio_projector(aud_gen_feats)
            
            else:
                t5_gen_feats = None
                aud_gen_feat = None

        except Exception as e:  # 捕获所有异常
            generated_reasoning = tokenizer.decode(generate_ids)
            t5_gen_feat = None
            aud_gen_feat = None
            caption_list = None
            generated_reasoning = None
            print('ERROR: ', e)
            pass

        return t5_gen_feat, aud_gen_feat, caption_list, generated_reasoning
        


    @classmethod
    def from_pretrained(cls, llm, input_resampler, output_resampler, whisper_resampler_llava, pretrained_model_path=None, **kwargs):
        model = cls(llm=llm, input_resampler=input_resampler, output_resampler=output_resampler, whisper_resampler_llava = whisper_resampler_llava, **kwargs)
        if os.environ.get('DEBUG_FLAG', 'False') == 'True':
            return model

        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            load_zero3_checkpoint(model, ckpt)
        return model