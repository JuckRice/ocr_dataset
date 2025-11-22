from __future__ import annotations
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import resize_pos_embed

class DinoV2Encoder(nn.Module):
    """
    Wrap timm's DINOv2 ViT as a visual encoder, returning patch tokens (excluding CLS).
    - model_name: e.g. "vit_base_patch14_dinov2.lvd142m"
    - out_dim: project to this dim for decoder cross-attention
    - freeze_stages: if >0, freeze earlier transformer blocks (0 means do not freeze blocks)
    """
    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m", out_dim: int = 512, freeze_stages: int = 0):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        assert hasattr(self.vit, "patch_embed") and hasattr(self.vit, "blocks")
        self.embed_dim = getattr(self.vit, "embed_dim", None) or getattr(self.vit, "num_features", None)
        self.out_proj = nn.Linear(self.embed_dim, out_dim) if out_dim != self.embed_dim else nn.Identity()
        self.freeze_stages = freeze_stages
        if hasattr(self.vit, "reset_classifier"):
            self.vit.reset_classifier(0)
        if freeze_stages > 0:
            for i, blk in enumerate(self.vit.blocks):
                if i < freeze_stages:
                    for p in blk.parameters():
                        p.requires_grad = False


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,3,H,W] with H,W multiples of the ViT patch size (e.g., 14).
        returns: patch tokens [B, T, D_out], where T = H_p * W_p.
        """
        # 调用 timm 内置的 forward_features 方法。
        # 这个方法会自动处理 patch_embed, 位置编码插值, 和所有 transformer 块。
        # 它返回所有token (包括 CLS token)。
        x = self.vit.forward_features(x)  # [B, T+1, D_model]

        # 移除 CLS Token (它总是在第0个位置)
        x = x[:, 1:, :]  # [B, T, D_model]

        # 将其投影到解码器所需的维度 (D_out)
        x = self.out_proj(x)

        return x  # [B, T, D_out]