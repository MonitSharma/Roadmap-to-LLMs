import torch
from torch.utils.data import Dataset
from typing import Any, Dict

def causal_mask(size: int) -> torch.Tensor:
    """Lower-triangular look-ahead mask.
    Returns shape [1, size, size] with True=keep and False=mask.
    """
    return torch.tril(torch.ones((1, size, size), dtype=torch.bool))

class BilingualDataset(Dataset):
    """Pairs of (src -> tgt) sequences for seq2seq/Transformer training.
    Expects each item in `ds` to be a dict with:
        item["translation"][src_lang] and item["translation"][tgt_lang]
    Tokenizers are HuggingFace `tokenizers.Tokenizer` instances.
    """
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang: str, tgt_lang: str, seq_len: int) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = int(seq_len)

        # Special tokens (target + source kept separate)
        # If your tokenizer uses different specials (e.g., <s>), change here.
        self.bos_src = self._tok_id_safe(self.tokenizer_src, "[BOS]", alt_tokens=["<s>", "<BOS>"])
        self.eos_src = self._tok_id_safe(self.tokenizer_src, "[EOS]", alt_tokens=["</s>", "<EOS>"])
        self.pad_src = self._tok_id_safe(self.tokenizer_src, "[PAD]", alt_tokens=["<pad>", "<PAD>"])

        self.bos_tgt = self._tok_id_safe(self.tokenizer_tgt, "[BOS]", alt_tokens=["<s>", "<BOS>"])
        self.eos_tgt = self._tok_id_safe(self.tokenizer_tgt, "[EOS]", alt_tokens=["</s>", "<EOS>"])
        self.pad_tgt = self._tok_id_safe(self.tokenizer_tgt, "[PAD]", alt_tokens=["<pad>", "<PAD>"])

    @staticmethod
    def _tok_id_safe(tok, primary: str, alt_tokens=None) -> int:
        tid = tok.token_to_id(primary)
        if tid is None and alt_tokens:
            for alt in alt_tokens:
                tid = tok.token_to_id(alt)
                if tid is not None:
                    break
        if tid is None:
            raise ValueError(f"Special token {primary} not found in tokenizer.")
        return tid

    def __len__(self) -> int:
        return len(self.ds)

    def _encode_src(self, text: str) -> torch.Tensor:
        ids = self.tokenizer_src.encode(text).ids
        seq = [self.bos_src] + ids + [self.eos_src]
        if len(seq) < self.seq_len:
            seq = seq + [self.pad_src] * (self.seq_len - len(seq))
        return torch.tensor(seq[: self.seq_len], dtype=torch.long)

    def _encode_tgt_inputs(self, text: str) -> torch.Tensor:
        ids = self.tokenizer_tgt.encode(text).ids
        seq = [self.bos_tgt] + ids + [self.eos_tgt]
        if len(seq) < self.seq_len:
            seq = seq + [self.pad_tgt] * (self.seq_len - len(seq))
        return torch.tensor(seq[: self.seq_len], dtype=torch.long)

    def _encode_tgt_labels(self, text: str) -> torch.Tensor:
        ids = self.tokenizer_tgt.encode(text).ids
        seq = ids + [self.eos_tgt]
        if len(seq) < self.seq_len:
            seq = seq + [self.pad_tgt] * (self.seq_len - len(seq))
        return torch.tensor(seq[: self.seq_len], dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]
        # HuggingFace translation datasets typically use this structure:
        if isinstance(item, dict) and "translation" in item:
            src_text = item["translation"][self.src_lang]
            tgt_text = item["translation"][self.tgt_lang]
        else:
            # Fallback: assume a tuple (src, tgt)
            src_text, tgt_text = item[0], item[1]

        encoder_input = self._encode_src(src_text)        # [S]
        decoder_input = self._encode_tgt_inputs(tgt_text) # [S]
        label         = self._encode_tgt_labels(tgt_text) # [S]

        # Masks (boolean): True=keep, False=mask
        # The encoder mask should have a shape of (1, 1, S) for a single item.
        # This allows it to be batched to (B, 1, 1, S) and broadcast correctly.
        encoder_mask = (encoder_input != self.pad_src).unsqueeze(0).unsqueeze(1) # [1,1,S] <-- FIX HERE

        # The decoder mask combines a look-ahead mask and a padding mask.
        look_ahead = causal_mask(self.seq_len)                                               # [1,S,S]
        key_padding = (decoder_input != self.pad_tgt).unsqueeze(0).unsqueeze(1) # [1,1,S] <-- FIX HERE
        decoder_mask = look_ahead & key_padding                                # [1,S,S]

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }