import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, drop=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)

        self.attn_drop = nn.Dropout(drop)      
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)      

    def forward(self, x):
        B, N, D = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)            

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)

        out = self.out_proj(out)
        out = self.proj_drop(out)       

        return out, attn
    

class MLP(nn.Module):
    def __init__(self, embed_dim=256, mlp_ratio=4, drop=0.1):
        super().__init__()

        hidden_dim = embed_dim * mlp_ratio

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(drop)       

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)                  
        x = self.fc2(x)
        x = self.dropout(x)                  
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4, drop=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, drop)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, drop)

        self.drop = nn.Dropout(drop)   # ✅ ADD

    def forward(self, x):
        # Attention
        attn_out, attn = self.attn(self.norm1(x))
        x = x + self.drop(attn_out)     # ✅ MODIFY

        # MLP
        x = x + self.drop(self.mlp(self.norm2(x)))   # ✅ MODIFY

        return x, attn
    
class TransformerEncoder(nn.Module):
    def __init__(self, depth=6, embed_dim=256, num_heads=8, mlp_ratio=4, drop=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])

    def forward(self, x):
        attn_maps = []

        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)

        return x, attn_maps