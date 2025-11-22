from __future__ import annotations
import io
from typing import List, Dict

PAD = "[PAD]"
BOS = "[BOS]"
EOS = "[EOS]"
UNK = "[UNK]"

SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

class Tokenizer:
    """
    Simple character-level tokenizer.
    - The vocab file has one token per line (UTF-8).
    - Special tokens [PAD], [BOS], [EOS], [UNK] are automatically added.
    """
    def __init__(self, vocab_path: str):
        with io.open(vocab_path, "r", encoding="utf-8") as f:
            tokens = [line.rstrip("\n") for line in f if line.strip() != ""]
        # De-duplicate while preserving order
        self.tokens = list(dict.fromkeys(tokens))
        # Ensure special tokens come first
        for sp in SPECIAL_TOKENS:
            if sp in self.tokens:
                self.tokens.remove(sp)
        self.tokens = SPECIAL_TOKENS + self.tokens
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.itos: Dict[int, str] = {i: t for t, i in self.stoi.items()}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in list(text)]
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], remove_special: bool = True) -> str:
        tokens = []
        for i in ids:
            if i in self.itos:
                t = self.itos[i]
                if remove_special and t in SPECIAL_TOKENS:
                    continue
                tokens.append(t)
        return "".join(tokens)

    def __len__(self):
        return len(self.tokens)
