from .blocks import LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        head_dim: int = 32,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        hidden_dim = heads * head_dim

        self.conv1 = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim * 3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.norm = LayerNorm(dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = input.shape
        q, k, v = torch.chunk(self.conv1(input), chunks=3, dim=1)

        q = q.view(batch_size, self.heads, self.head_dim, height * width)
        k = k.view(batch_size, self.heads, self.head_dim, height * width)
        v = v.view(batch_size, self.heads, self.head_dim, height * width)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (height * width)

        context = torch.matmul(k, v.transpose(-2, -1))

        out = torch.matmul(context.transpose(-2, -1), q)
        out = out.view(
            batch_size,
            self.heads,
            self.head_dim,
            height,
            width
        )
        out = out.reshape(
            batch_size,
            self.heads * self.head_dim,
            height,
            width
        )
        out = self.conv2(out)
        out = self.norm(out)

        return out

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        head_dim: int = 32,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        hidden_dim = heads * head_dim

        self.conv1 = nn.Conv2d(
            in_channels=dim,
            out_channels=hidden_dim * 3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = input.shape
        q, k, v = torch.chunk(self.conv1(input), chunks=3, dim=1)

        q = q.view(batch_size, self.heads, self.head_dim, height * width)
        k = k.view(batch_size, self.heads, self.head_dim, height * width)
        v = v.view(batch_size, self.heads, self.head_dim, height * width)

        q = F.normalize(
            q,
            dim=-1,
            eps=1e-6,
        )
        k = F.normalize(
            k,
            dim=-1,
            eps=1e-6,
        )

        attn = torch.matmul(q.transpose(2, 3), k)
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(2, 3)
        out = out.view(
            batch_size,
            self.heads,
            self.head_dim,
            height,
            width
        )
        out = out.reshape(
            batch_size,
            self.heads * self.head_dim,
            height,
            width
        )
        out = self.conv2(out)
        return out
