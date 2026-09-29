import torch
import torch.nn as nn

import math

class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(
                half_dim,
                device=input.device,
            ) * -embeddings
        )
        embeddings = input[:, None] * embeddings[None, :]
        embeddings = torch.cat(
            (
                embeddings.sin(),
                embeddings.cos(),
            ),
            dim=-1
        )
        return embeddings
