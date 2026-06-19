import json
import logging
import os
import numpy as np
from typing import List, Union, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)

class CharTokenizer:
    """Simple character-level tokenizer for CTC-based Russian speech recognition."""
    def __init__(self, blank_token="<blank>", pad_token="<pad>", unk_token="<unk>"):
        self.blank_token = blank_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "
        self.vocab = [blank_token, pad_token, unk_token] + list(alphabet)
        self.char2id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id2char = {idx: char for char, idx in self.char2id.items()}
        self.blank_token_id = self.char2id[blank_token]
        self.pad_token_id = self.char2id[pad_token]
        self.unk_token_id = self.char2id[unk_token]

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text: Union[str, List[str]], padding=False, return_tensors=None):
        if isinstance(text, str):
            text = [text]
        
        encoded = []
        max_len = 0
        for s in text:
            s = s.lower()
            ids = [self.char2id.get(c, self.unk_token_id) for c in s]
            encoded.append(ids)
            max_len = max(max_len, len(ids))
        
        attention_mask = []
        if padding:
            for i in range(len(encoded)):
                mask = [1] * len(encoded[i])
                pads_needed = max_len - len(encoded[i])
                encoded[i].extend([self.pad_token_id] * pads_needed)
                mask.extend([0] * pads_needed)
                attention_mask.append(mask)
        else:
            attention_mask = [[1] * len(seq) for seq in encoded]

        res = {"input_ids": encoded, "attention_mask": attention_mask}
        
        if return_tensors == "pt":
            res["input_ids"] = torch.tensor(res["input_ids"], dtype=torch.long)
            res["attention_mask"] = torch.tensor(res["attention_mask"], dtype=torch.long)
            
        return res

    def decode(self, ids: List[int]) -> str:
        return "".join([self.id2char.get(i, self.unk_token) for i in ids])

    def batch_decode(self, sequences: List[List[int]], skip_special_tokens=True) -> List[str]:
        res = []
        for seq in sequences:
            if hasattr(seq, "tolist"):
                seq = seq.tolist()
            if skip_special_tokens:
                seq = [i for i in seq if i not in [self.blank_token_id, self.pad_token_id, self.unk_token_id]]
            res.append(self.decode(seq))
        return res

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

class CustomProcessor:
    """Wrapper that combines audio feature extraction and text tokenization."""
    def __init__(self, tokenizer, sample_rate=16000, n_mels=80):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160
        )

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        if len(args) > 0 and (isinstance(args[0], str) or (isinstance(args[0], list) and isinstance(args[0][0], str))):
            return self.tokenizer(*args, **kwargs)
        if "text" in kwargs:
            return self.tokenizer(text=kwargs.pop("text"), **kwargs)
            
        audio = args[0] if len(args) > 0 else kwargs.get("audio")
        if audio is not None:
            return self._process_audio(audio, **kwargs)
            
        raise ValueError("Нужно передать либо текст, либо аудио.")

    def _process_audio(self, audio, sampling_rate=16000, return_tensors="pt", **kwargs):
        if isinstance(audio, (list, np.ndarray)):
            waveform = torch.tensor(audio).float()
        elif torch.is_tensor(audio):
            waveform = audio.float()
        else:
            waveform = torch.tensor(audio).float()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        mel = self.mel_transform(waveform)
        mel = torch.log(mel + 1e-9)
        mel = (mel - mel.mean(dim=-1, keepdim=True)) / (mel.std(dim=-1, keepdim=True) + 1e-9)
        
        if return_tensors == "pt":
            return {"input_features": mel}
        return {"input_features": mel.numpy()}

    def batch_decode(self, sequences, skip_special_tokens=True):
        res = []
        for seq in sequences:
            if hasattr(seq, "tolist"):
                seq = seq.tolist()
            chars = []
            prev_char = -1
            for char_id in seq:
                if char_id != prev_char and char_id != self.tokenizer.blank_token_id:
                    if not (skip_special_tokens and char_id in [self.tokenizer.pad_token_id, self.tokenizer.unk_token_id]):
                        chars.append(char_id)
                prev_char = char_id
            res.append(self.tokenizer.decode(chars))
        return res
        
    def save_pretrained(self, save_directory):
        self.tokenizer.save_pretrained(save_directory)

class CustomCNNBiGRU(nn.Module):
    """Acoustic model combining 2D convolutions, BiGRU, and CTC Loss."""
    def __init__(self, n_mels: int, vocab_size: int, hidden_size: int = 256, num_gru_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        cnn_out_freq = n_mels // 4 
        rnn_input_size = 64 * cnn_out_freq
        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_gru_layers > 1 else 0
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, input_features, input_lengths=None, labels=None, target_lengths=None):
        x = input_features.unsqueeze(1)
        x = self.cnn(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, t, c * f)
        x, _ = self.rnn(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            if input_lengths is not None:
                out_lengths = input_lengths // 4
            else:
                out_lengths = torch.full(size=(b,), fill_value=t, dtype=torch.long, device=logits.device)
            if target_lengths is None:
                target_lengths = torch.sum(labels != -100, dim=1)
            labels_flat = labels[labels != -100]
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = self.ctc_loss(log_probs, labels_flat, out_lengths, target_lengths)
            
        return {"loss": loss, "logits": logits}

def build_custom_model(cfg: Dict[str, Any]) -> Tuple[CustomCNNBiGRU, CustomProcessor]:
    """Initializes the custom acoustic model and processor based on config."""
    tokenizer = CharTokenizer()
    vocab_size = len(tokenizer)
    n_mels = cfg['model'].get('n_mels', 80)
    
    model = CustomCNNBiGRU(
        n_mels=n_mels,
        vocab_size=vocab_size,
        hidden_size=cfg['model'].get('hidden_size', 256),
        num_gru_layers=cfg['model'].get('num_gru_layers', 2)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Custom Model parameters: total=%d", total_params)
    
    processor = CustomProcessor(tokenizer, n_mels=n_mels)
    return model, processor

def _custom_generate(model, input_features, processor=None, input_lengths=None):
    with torch.no_grad():
        outputs = model(input_features, input_lengths=input_lengths)
        logits = outputs["logits"]
        return torch.argmax(logits, dim=-1)