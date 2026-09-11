import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=256,
        stride=None   # for overlapping patches
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        # if stride not given → non-overlapping
        self.stride = stride if stride is not None else patch_size

        # conv does patch extraction + projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=self.stride
        )

        # number of patches (important!)
        self.num_patches = (
            (img_size - patch_size) // self.stride + 1
        ) ** 2

    def forward(self, x):
        """
        x: [B, 3, 32, 32]
        """

        x = self.proj(x)  
        # → [B, embed_dim, H', W']

        x = x.flatten(2)  
        # → [B, embed_dim, N]

        x = x.transpose(1, 2)  
        # → [B, N, embed_dim]

        return x