from __future__ import annotations
import io
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from .transforms import default_train_transform, default_val_transform
from ..utils.tokenizer import Tokenizer

class OCRDataset(Dataset):
    def __init__(self, ann_path: str, tokenizer: Tokenizer, img_height: int, max_width: int,
                 patch_size: int = 14, is_train: bool = True):
        """
        Generic OCR dataset that reads a TSV/CSV-like annotation file.
        Each line should be:  path<TAB>text  (UTF-8). If no tab is found, a single comma (",") is tried.
        Otherwise, whitespace splitting is used as a last resort (first field is path, the rest is text).
        """
        self.samples: List[Tuple[str, str]] = []
        with io.open(ann_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:
                    continue
                if "\t" in line:
                    # Literal backslash+t present; treat it as delimiter
                    p, t = line.split("\t", 1)
                else:
                    # Prefer a real tab if present
                    if "\t" not in line:
                        pass  # fall-through below
                    # Use actual TAB if any, else comma, else whitespace
                    if "\t" not in line and "," not in line:
                        parts = line.split()
                        p, t = parts[0], "".join(parts[1:]) if len(parts) > 1 else ""
                    elif "\t" not in line and "," in line:
                        p, t = line.split(",", 1)
                    else:
                        p, t = line.split("\t", 1)  # fallback
                self.samples.append((p, t))

        self.tokenizer = tokenizer
        self.img_height = img_height
        self.max_width = max_width
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, text = self.samples[idx]
        img = Image.open(path)
        if self.is_train:
            x = default_train_transform(img, self.img_height, self.max_width, self.patch_size)
        else:
            x = default_val_transform(img, self.img_height, self.max_width, self.patch_size)
        ids = self.tokenizer.encode(text, add_special=True)
        return {"image": x, "target_ids": ids, "text": text, "path": path}

def ocr_collate_fn(batch, pad_id: int, max_tgt_len: int = None):
    import torch
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]
    paths = [b["path"] for b in batch]
    ids_list = [b["target_ids"] for b in batch]

    # Pad targets to the same length
    if max_tgt_len is None:
        L = max(len(x) for x in ids_list)
    else:
        L = min(max(len(x) for x in ids_list), max_tgt_len)

    tgt = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        ids = ids[:L]  # truncate if needed
        tgt[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    images = torch.stack(images, dim=0).float()
    return {"images": images, "tgt": tgt, "texts": texts, "paths": paths}
