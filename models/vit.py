import torch
import torch.nn as nn

from models.patch_embedding import PatchEmbedding
from models.transformer import TransformerEncoder


class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        use_cls_token=True,
        pos_embed_type="learnable",  # "learnable", "sinusoidal", "none"
        stride=None  # for overlapping patches
    ):
        super().__init__()

        self.use_cls_token = use_cls_token
        self.pos_embed_type = pos_embed_type

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim, stride
        )

        num_patches = self.patch_embed.num_patches

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            num_patches += 1
        else:
            self.cls_token = None

        # Positional encoding
        if pos_embed_type == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        elif pos_embed_type == "sinusoidal":
            self.pos_embed = self._build_sinusoidal_pos_embed(num_patches, embed_dim)
        else:
            self.pos_embed = None

        # Transformer
        self.encoder = TransformerEncoder(depth, embed_dim, num_heads, mlp_ratio,drop=0.1)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def _build_sinusoidal_pos_embed(self, num_patches, dim):
        pe = torch.zeros(num_patches, dim)

        position = torch.arange(0, num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, N, D]

    def forward(self, x):
        """
        x: [B, 3, 32, 32]
        """

        x = self.patch_embed(x)  # [B, N, D]

        B, N, D = x.shape

        # Add CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, D]

        # Add positional encoding
        if self.pos_embed is not None:
            x = x + self.pos_embed.to(x.device)

        # Transformer
        x, attn_maps = self.encoder(x)

        # Pooling
        if self.use_cls_token:
            x = x[:, 0]  # CLS token
        else:
            x = x.mean(dim=1)  # mean pooling

        # Classification
        out = self.head(x)

        return out, attn_maps