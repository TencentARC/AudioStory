import os
import torch
import transformers
from transformers import Qwen2ForCausalLM
import torch


def load_checkpoint(source_model, target_model):
    state_dict = torch.load(source_model)
    target_model = transformers.AutoModelForCausalLM.from_pretrained(target_model)
    target_model.load_state_dict(state_dict, strict=True)
    print('init qwen llm weight done.')

    return target_model