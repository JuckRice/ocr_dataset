from __future__ import annotations
import math
import torch
import torch.nn as nn
from .dinov2_encoder import DinoV2Encoder

def sinusoidal_positional_encoding(length: int, d_model: int, device=None):
    """Classic sinusoidal positional encoding (1D)."""
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [L, D]

def generate_square_subsequent_mask(sz: int, device=None):
    """Causal mask for autoregressive decoding (True means masked/ignored)."""
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

class OCRTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        d_model: int = 512,
        nhead: int = 8,
        dec_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        freeze_stages: int = 0
    ):
        super().__init__()
        self.encoder = DinoV2Encoder(model_name=model_name, out_dim=d_model, freeze_stages=freeze_stages)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_drop = nn.Dropout(dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, images: torch.Tensor, tgt_tokens: torch.Tensor) -> dict:
        """
        images: [B,3,H,W]
        tgt_tokens: [B, T] containing BOS...target...EOS...PAD (PAD id must be 0 with the provided tokenizer)
        returns: {"logits": [B, T, V]}
        """
        device = images.device
        # Visual memory from DINOv2 encoder
        memory = self.encoder(images)  # [B, S, D]

        # Target embeddings + positional encodings
        B, T = tgt_tokens.size()
        tok = self.token_embed(tgt_tokens)  # [B, T, D]
        pe = sinusoidal_positional_encoding(T, tok.size(-1), device=device).unsqueeze(0)  # [1,T,D]
        tok = self.pos_drop(tok + pe)

        tgt_mask = generate_square_subsequent_mask(T, device=device)  # [T,T]
        tgt_padding_mask = (tgt_tokens == 0)  # assumes PAD id == 0

        dec_out = self.decoder(
            tgt=tok,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )  # [B, T, D]
        dec_out = self.norm(dec_out)
        logits = self.out(dec_out)  # [B, T, V]
        return {"logits": logits}
