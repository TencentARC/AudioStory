import torch
from transformers import LogitsProcessor


BOA_TOKEN = '<aud>'
EOA_TOKEN = '</aud>'
AUD_TOKEN = '<aud_{:05d}>'

class AutoAudioTokenGenerationProcessor(LogitsProcessor):

    def __init__(self, tokenizer, num_aud_gen_tokens=32) -> None:
        super().__init__()
        # self.boi_token_id = tokenizer.encode(BOA_TOKEN)[0]
        # self.eoi_token_id = tokenizer.encode(EOA_TOKEN)[0]
        aud_all_token_str = ''.join([BOA_TOKEN] + [AUD_TOKEN.format(int(item))
                                                   for item in range(num_aud_gen_tokens)] + [EOA_TOKEN])
        self.aud_ids_list = tokenizer.encode(aud_all_token_str, add_special_tokens=False)

    def __call__(self, input_ids, scores):
        bz = input_ids.shape[0]
        for i in range(bz):
            cur_input_id = input_ids[i, -1].item()
            if cur_input_id in self.aud_ids_list[:-1]:

                output_id = self.aud_ids_list[self.aud_ids_list.index(cur_input_id) + 1]
                scores[i, ..., output_id] = scores[i, ...].max() + 10.
            else:

                scores[i, ..., torch.tensor(self.aud_ids_list[1:]).to(dtype=torch.long)] = 0.0

        return scores



BOT_TOKEN = '<t5>'
EOT_TOKEN = '</t5>'
T5_TOKEN = '<t5_{:05d}>'

class AutoT5TokenGenerationProcessor(LogitsProcessor):

    def __init__(self, tokenizer, num_t5_gen_tokens=64) -> None:
        super().__init__()
        # self.boi_token_id = tokenizer.encode(BOA_TOKEN)[0]
        # self.eoi_token_id = tokenizer.encode(EOA_TOKEN)[0]
        t5_all_token_str = ''.join([BOT_TOKEN] + [T5_TOKEN.format(int(item))
                                                   for item in range(num_t5_gen_tokens)] + [EOT_TOKEN])
        self.t5_ids_list = tokenizer.encode(t5_all_token_str, add_special_tokens=False)

    def __call__(self, input_ids, scores):
        bz = input_ids.shape[0]
        for i in range(bz):
            cur_input_id = input_ids[i, -1].item()
            if cur_input_id in self.t5_ids_list[:-1]:

                output_id = self.t5_ids_list[self.t5_ids_list.index(cur_input_id) + 1]
                scores[i, ..., output_id] = scores[i, ...].max() + 10.
            else:

                scores[i, ..., torch.tensor(self.t5_ids_list[1:]).to(dtype=torch.long)] = 0.0

        return scores
