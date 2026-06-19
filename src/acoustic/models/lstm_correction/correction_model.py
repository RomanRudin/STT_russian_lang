import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CharTokenizer:
    """Character-level tokenizer for Russian text correction."""

    def __init__(self, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>"):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя -.,!?\"'"
        self.vocab = [pad_token, bos_token, eos_token, unk_token] + list(self.alphabet)
        self.char2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2char = {i: c for c, i in self.char2id.items()}
        self.pad_token_id = self.char2id[pad_token]
        self.bos_token_id = self.char2id[bos_token]
        self.eos_token_id = self.char2id[eos_token]
        self.unk_token_id = self.char2id[unk_token]

    def __len__(self):
        return len(self.vocab)

    def encode(self, text: str) -> list:
        text = text.lower().strip()
        return [self.char2id.get(c, self.unk_token_id) for c in text]

    def decode(self, ids: list) -> str:
        return "".join([self.id2char.get(i, self.unk_token) for i in ids
                        if i not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]])

    def batch_decode(self, sequences, skip_special_tokens=True):
        res = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            if skip_special_tokens:
                seq = [i for i in seq if i not in
                       [self.pad_token_id, self.bos_token_id, self.eos_token_id]]
            res.append(self.decode(seq))
        return res

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        vocab_path = os.path.join(save_directory, "vocab.json")
        import json
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory):
        import json
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            char2id = json.load(f)
        tokenizer = cls()
        tokenizer.char2id = char2id
        tokenizer.id2char = {i: c for c, i in char2id.items()}
        tokenizer.pad_token_id = char2id[tokenizer.pad_token]
        tokenizer.bos_token_id = char2id[tokenizer.bos_token]
        tokenizer.eos_token_id = char2id[tokenizer.eos_token]
        tokenizer.unk_token_id = char2id[tokenizer.unk_token]
        return tokenizer
    
    def __call__(self, text, return_tensors=None, truncation=False, padding=False, max_length=None):
        if isinstance(text, str):
            text = [text]
        ids = [self.encode(t) for t in text]
        if truncation and max_length:
            ids = [i[:max_length] for i in ids]
        if padding:
            max_len = max_length if max_length else max(len(i) for i in ids)
            ids = [i + [self.pad_token_id] * (max_len - len(i)) for i in ids]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids, dtype=torch.long)}
        return {"input_ids": ids}


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs):
        energy = self.Va(torch.tanh(self.Wa(encoder_outputs)))
        weights = F.softmax(energy.squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1)


class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def forward(self, tgt, encoder_outputs, hidden, cell):
        embedded = self.dropout(self.embedding(tgt))
        context = self.attention(encoder_outputs)
        context = context.unsqueeze(1).repeat(1, tgt.size(1), 1)
        lstm_input = torch.cat([embedded, context], dim=2)
        outputs, _ = self.lstm(lstm_input, (hidden, cell))
        logits = self.fc(outputs)
        return logits


class LSTMSeq2Seq(nn.Module):
    """Character-level LSTM sequence-to-sequence model with attention."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=2, dropout=0.25):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout

        self.encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(vocab_size, embedding_dim, hidden_size * 2, num_layers, dropout)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def _prepare_decoder_init_state(self, encoder_hidden, encoder_cell):
        num_layers = self.num_layers
        batch_size = encoder_hidden.size(1)

        def merge_directions(state):
            state = state.view(num_layers, 2, batch_size, self.hidden_size)
            return torch.cat([state[:, 0], state[:, 1]], dim=-1)

        return merge_directions(encoder_hidden), merge_directions(encoder_cell)

    def forward(self, input_ids, labels=None, **kwargs):
        encoder_outputs, enc_hidden, enc_cell = self.encoder(input_ids)
        dec_hidden, dec_cell = self._prepare_decoder_init_state(enc_hidden, enc_cell)

        if labels is not None:
            decoder_input = labels[:, :-1]
            logits = self.decoder(decoder_input, encoder_outputs, dec_hidden, dec_cell)
            loss = self.loss_fn(logits.reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1))
            return {"loss": loss, "logits": logits}
        else:
            batch_size = input_ids.size(0)
            device = input_ids.device
            decoder_input = torch.tensor([[1]], device=device).repeat(batch_size, 1)
            max_len = 220
            outputs = []
            hidden, cell = dec_hidden, dec_cell
            context = self.decoder.attention(encoder_outputs)

            for _ in range(max_len):
                embedded = self.decoder.embedding(decoder_input)
                lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
                output, (hidden, cell) = self.decoder.lstm(lstm_input, (hidden, cell))
                prediction = self.decoder.fc(output.squeeze(1))
                token = prediction.argmax(dim=1)
                outputs.append(token.unsqueeze(1))
                decoder_input = token.unsqueeze(1)
                if (token == 2).all():
                    break
            logits = torch.cat(outputs, dim=1)
            return {"logits": logits}

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), path)
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        }
        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, save_directory, **kwargs):
        import json
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model


def build_lstm_correction(cfg: Dict[str, Any]) -> Tuple[LSTMSeq2Seq, CharTokenizer]:
    tokenizer = CharTokenizer()
    vocab_size = len(tokenizer)

    hidden_size = cfg['model'].get('hidden_size', 256)
    num_layers = cfg['model'].get('num_layers', 2)
    dropout = cfg['model'].get('dropout', 0.25)
    embedding_dim = cfg['model'].get('embedding_dim', 128)

    model = LSTMSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    logger.info("LSTM correction model created: vocab=%d, hidden=%d, layers=%d",
                vocab_size, hidden_size, num_layers)
    return model, tokenizer


def _lstm_generate(model, input_data, processor):
    model.eval()
    with torch.no_grad():
        outputs = model(input_data)
        return outputs["logits"]


def load_lstm_checkpoint(checkpoint_dir: str, cfg: Dict[str, Any]) -> Tuple[LSTMSeq2Seq, CharTokenizer]:
    logger.info("Loading LSTM checkpoint from %s", checkpoint_dir)
    tokenizer = CharTokenizer.from_pretrained(checkpoint_dir)
    model = LSTMSeq2Seq.from_pretrained(checkpoint_dir)
    return model, tokenizer