import torch
from typing import Dict, Any, List

class LSTMCorrectionCollator:
    def __init__(self, tokenizer, max_length=256, augmentations=None, **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        src_texts = [f["stt_text"] for f in features]
        tgt_texts = [f["reference"] for f in features]

        src_ids = []
        tgt_ids = []
        for src, tgt in zip(src_texts, tgt_texts):
            src_enc = self.tokenizer.encode(src)
            if len(src_enc) > self.max_length:
                src_enc = src_enc[:self.max_length]
            tgt_enc = [self.tokenizer.bos_token_id] + self.tokenizer.encode(tgt) + [self.tokenizer.eos_token_id]
            if len(tgt_enc) > self.max_length:
                tgt_enc = tgt_enc[:self.max_length - 1] + [self.tokenizer.eos_token_id]
            src_ids.append(src_enc)
            tgt_ids.append(tgt_enc)

        src_max_len = max(len(s) for s in src_ids)
        tgt_max_len = max(len(t) for t in tgt_ids)
        src_padded = torch.zeros(len(src_ids), src_max_len, dtype=torch.long).fill_(self.tokenizer.pad_token_id)
        tgt_padded = torch.zeros(len(tgt_ids), tgt_max_len, dtype=torch.long).fill_(self.tokenizer.pad_token_id)

        for i, (s, t) in enumerate(zip(src_ids, tgt_ids)):
            src_padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            tgt_padded[i, :len(t)] = torch.tensor(t, dtype=torch.long)

        return {"input_ids": src_padded, "labels": tgt_padded}