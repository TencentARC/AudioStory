from transformers import AutoModelForCausalLM, AutoTokenizer, AddedToken
import torch
import json


def init_tokenizer(pretrained_model_path, add_tokens_path=None):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    if add_tokens_path:
        with open(add_tokens_path, 'r') as f:
            add_tokens = json.load(f)
        add_tokens = list(add_tokens.keys())
        special_tokens = [
            AddedToken(
                t, 
                special=True
            ) for t in add_tokens
        ]
        
        tokenizer.add_special_tokens({
            "additional_special_tokens": [t.content for t in special_tokens]
        })

    return tokenizer