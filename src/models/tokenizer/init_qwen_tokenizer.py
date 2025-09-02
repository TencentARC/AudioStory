from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json


def init_tokenizer(pretrained_model_path, add_tokens_path=None):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    if add_tokens_path:
        with open(add_tokens_path, 'r') as f:
            add_tokens = json.load(f)
        add_tokens = list(add_tokens.keys())
        tokens_number = tokenizer.add_tokens(add_tokens)

    return tokenizer