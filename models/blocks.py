import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight

        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), keepdim=True, unbiased=False)

        normalized_weight = (weight - mean) * torch.rsqrt(var + 1e-5)
        normalized_weight = normalized_weight

        bias = self.bias

        out = F.conv2d(
            input,
            normalized_weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return out

class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int=8,
    ) -> None:
        super().__init__()
        self.conv = WeightStandardizedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm = nn.GroupNorm(groups, out_channels, eps=1e-5)
        self.act = nn.SiLU()

    def forward(
            self,
            input: torch.Tensor,
            scale_shift: tuple[torch.Tensor, torch.Tensor]=None,
        ) -> torch.Tensor:
        out = self.conv(input)
        out = self.norm(out)
        if scale_shift is not None:
            scale, shift = scale_shift
            scale = scale.to(device=out.device)
            shift = shift.to(device=out.device)

            scale = scale.clamp(min=-5, max=5)
            shift = shift.clamp(min=-5, max=5)

            out = out * (scale + 1) + shift
        out = self.act(out)
        return out

class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_channels: int,
        time_embedding_dim: int,
        groups: int=8,
    ) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, 2 * out_channels)
        )
        self.block1 = Block(in_dim, out_channels, groups=groups)
        self.block2 = Block(out_channels, out_channels, groups=groups)
        self.conv = nn.Conv2d(in_dim, out_channels, 1) if in_dim != out_channels else nn.Identity()

    def forward(
            self,
            input: torch.Tensor,
            time_embeddings: torch.Tensor=None,
        ) -> torch.Tensor:
        scale_shift = None
        if time_embeddings is not None:
            time_embeddings = self.ff(time_embeddings)
            time_embeddings = time_embeddings.view(time_embeddings.size(0), time_embeddings.size(1), 1, 1)
            scale_shift = time_embeddings.chunk(2, dim = 1)
        out = self.block1(input, scale_shift=scale_shift)
        out = self.block2(out)
        out = out + self.conv(input)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.upsample(input)
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.downsample(input)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input
        var = torch.var(input, dim=1, unbiased=False, keepdim=True)
        mu = torch.mean(input, dim=1, keepdim=True)
        out = (input - mu) * torch.rsqrt(var + 1e-5) * self.g
        return out
