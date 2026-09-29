from .blocks import ResNetBlock, DownsampleBlock, UpsampleBlock, LayerNorm
from .attention import LinearAttention, Attention
from .embeddings import SinusoidalPositionalEmbeddings

import numpy as np
import torch
import torch.nn as nn

import torchutils

class UNet(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: tuple[int, ...]=[64, 128, 256, 512],
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(
            in_channels=self.img_size[0],
            out_channels=self.in_channels[0],
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.time_embedding_dim = 4 * self.in_channels[0]

        self.ff = nn.Sequential(
            SinusoidalPositionalEmbeddings(self.in_channels[0]),
            nn.Linear(self.in_channels[0], self.time_embedding_dim),
            nn.GELU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )
        
        self.downsample = nn.ModuleList([])
        for idx in range(len(self.in_channels) - 2):
            self.downsample.append(nn.ModuleList([
                ResNetBlock(
                    in_dim=self.in_channels[idx],
                    out_channels=self.in_channels[idx],
                    time_embedding_dim=self.time_embedding_dim,
                    groups=8,
                ),
                ResNetBlock(
                    in_dim=self.in_channels[idx],
                    out_channels=self.in_channels[idx],
                    time_embedding_dim=self.time_embedding_dim,
                    groups=8,
                ),
                LayerNorm(self.in_channels[idx]),
                LinearAttention(
                    dim=self.in_channels[idx],
                    heads=4,
                    head_dim=32,
                ),
                DownsampleBlock(
                    in_channels=self.in_channels[idx],
                    out_channels=self.in_channels[idx + 1],
                )
            ]))
        idx += 1
        self.downsample.append(nn.ModuleList([
            ResNetBlock(
                in_dim=self.in_channels[idx],
                out_channels=self.in_channels[idx],
                time_embedding_dim=self.time_embedding_dim,
                groups=8,
            ),
            ResNetBlock(
                in_dim=self.in_channels[idx],
                out_channels=self.in_channels[idx],
                time_embedding_dim=self.time_embedding_dim,
                groups=8,
            ),
            LayerNorm(self.in_channels[idx]),
            LinearAttention(
                dim=self.in_channels[idx],
                heads=4,
                head_dim=32
            ),
            nn.Conv2d(
                in_channels=self.in_channels[idx],
                out_channels=self.in_channels[idx + 1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        ]))

        self.middle_blocks = nn.ModuleList([
            ResNetBlock(
                    in_dim=self.in_channels[-1],
                    out_channels=self.in_channels[-1],
                    time_embedding_dim=self.time_embedding_dim,
                    groups=8,
            ),
            LayerNorm(self.in_channels[-1]),
            Attention(
                dim=self.in_channels[-1],
                heads=4,
                head_dim=32,
            ),
            ResNetBlock(
                    in_dim=self.in_channels[-1],
                    out_channels=self.in_channels[-1],
                    time_embedding_dim=self.time_embedding_dim,
                    groups=8,
            ),
        ])

        self.upsample = nn.ModuleList([])
        for idx in reversed(range(1, len(in_channels) - 1)):
            self.upsample.append(nn.ModuleList([
                ResNetBlock(
                    in_dim=self.in_channels[idx] + self.in_channels[idx + 1],
                    out_channels=self.in_channels[idx + 1],
                    time_embedding_dim=self.time_embedding_dim,
                    groups=8,
                ),
                ResNetBlock(
                    in_dim=self.in_channels[idx] + self.in_channels[idx + 1],
                    out_channels=self.in_channels[idx + 1],
                    time_embedding_dim=self.time_embedding_dim,
                    groups=8,
                ),
                LayerNorm(self.in_channels[idx + 1]),
                LinearAttention(
                    dim=self.in_channels[idx + 1],
                    heads=4,
                    head_dim=32,
                ),
                UpsampleBlock(
                    in_channels=self.in_channels[idx + 1],
                    out_channels=self.in_channels[idx],
                )
            ]))
        idx -= 1
        self.upsample.append(nn.ModuleList([
            ResNetBlock(
                in_dim=self.in_channels[idx] + self.in_channels[idx + 1],
                out_channels=self.in_channels[idx + 1],
                time_embedding_dim=self.time_embedding_dim,
                groups=8,
            ),
            ResNetBlock(
                in_dim=self.in_channels[idx] + self.in_channels[idx + 1],
                out_channels=self.in_channels[idx + 1],
                time_embedding_dim=self.time_embedding_dim,
                groups=8,
            ),
            LayerNorm(self.in_channels[idx + 1]),
            LinearAttention(
                dim=self.in_channels[idx + 1],
                heads=4,
                head_dim=32,
            ),
            nn.Conv2d(
                in_channels=self.in_channels[idx + 1],
                out_channels=self.in_channels[idx],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        ]))

        self.resnet_block = ResNetBlock(
            in_dim=2 * self.in_channels[0],
            out_channels=self.in_channels[0],
            time_embedding_dim=self.time_embedding_dim,
            groups=8,
        )

        self.conv2 = nn.Conv2d(
            in_channels=self.in_channels[0],
            out_channels=self.img_size[0],
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, input: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        out = self.conv1(input)
        residual = out
        t = self.ff(time)

        h = []
        for resnet_block1, resnet_block2, norm_block, attention_block, downsample_block in self.downsample:
            out = resnet_block1(out, t)
            h.append(out)

            out = resnet_block2(out, t)
            h.append(out)
            
            norm_out = norm_block(out)
            out = attention_block(norm_out) + norm_out
            out = downsample_block(out)

        resnet_block1, norm_block, attention_block, resnet_block2 = self.middle_blocks
        out = resnet_block1(out, t)
        norm_out = norm_block(out)
        out = attention_block(norm_out) + norm_out
        out = resnet_block2(out, t)

        for resnet_block1, resnet_block2, norm_block, attention_block, upsample_block in self.upsample:
            out = torch.cat([out, h.pop()], dim=1)
            out = resnet_block1(out, t)

            out = torch.cat([out, h.pop()], dim=1)
            out = resnet_block2(out, t)
            
            norm_out = norm_block(out)
            out = attention_block(norm_out) + norm_out
            out = upsample_block(out)

        out = torch.cat([out, residual], dim=1)
        out = self.resnet_block(out, t)
        out = self.conv2(out)
        return out

def cosine_beta_schedule(timesteps: int, s: float=0.008):
    total_steps = timesteps + 1
    x = torch.linspace(0, timesteps, total_steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0, 0.999)
    return betas

class DDPM(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: tuple[int, ...]=[64, 128, 256, 512],
        timesteps: int=1000,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.unet = UNet(
            img_size=img_size,
            in_channels=in_channels
        )
        betas = cosine_beta_schedule(timesteps, s=0.008)
        self.num_timesteps = betas.shape[0]

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([
            torch.ones(1, dtype=alphas_cumprod.dtype),
            alphas_cumprod[:-1]
        ])

        x0_coefficient1 = 1. / torch.sqrt(alphas_cumprod)
        x0_coefficient2 = -x0_coefficient1 * torch.sqrt(1. - alphas_cumprod)

        posterior_mean_coefficient1 = (torch.sqrt(alphas_cumprod_prev) * betas) / (1. - alphas_cumprod)
        posterior_mean_coefficient2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev)) / (1. - alphas_cumprod)

        posterior_variance =  ((1. - alphas_cumprod_prev) / (1 - alphas_cumprod)) * betas

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("x0_coefficient1", x0_coefficient1)
        self.register_buffer("x0_coefficient2", x0_coefficient2)
        self.register_buffer("posterior_mean_coefficient1", posterior_mean_coefficient1)
        self.register_buffer("posterior_mean_coefficient2", posterior_mean_coefficient2)
        self.register_buffer("posterior_variance", posterior_variance)

    def forward(self, input: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        predicted_noise = self.unet(input, time)
        mu = self.x0_coefficient1.gather(
            0,
            time
        ).reshape(
            time.shape[0], *((1,) * (len(input.shape) - 1))
        )
        var = self.x0_coefficient2.gather(
            0,
            time
        ).reshape(
            time.shape[0], *((1,) * (len(input.shape) - 1))
        )
        x0 = (
            mu * input
            + var * predicted_noise
        ).clamp(min=-1, max=1)
        out = (predicted_noise, x0)
        return out

    @torch.no_grad()
    def generate_samples(self, num_samples: int, dtype: str) -> torch.Tensor:
        device = next(self.parameters()).device
        dtype = torchutils.get_torch_dtype(dtype)
        use_amp = (device.type == "cuda") and (dtype != torch.float32)
        self.unet.eval()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            z = torch.randn(
                (num_samples, *self.img_size),
                device=device,
                dtype=dtype,
            )
            for t in range(self.num_timesteps - 1, 0, -1):
                timesteps = torch.full(
                    (z.shape[0],),
                    t,
                    dtype=torch.long,
                    device=device,
                )
                _, x0 = self.forward(z, timesteps)
                z = z.to(dtype)
                x0 = x0.to(dtype)
                posterior_mu = self.posterior_mean_coefficient1.gather(
                    -1,
                    timesteps
                ).reshape(
                    timesteps.shape[0], *((1,) * (len(z.shape) - 1))
                ) * x0
                posterior_mu = posterior_mu + self.posterior_mean_coefficient2.gather(
                    -1,
                    timesteps
                ).reshape(
                    timesteps.shape[0], *((1,) * (len(z.shape) - 1))
                ) * z

                posterior_var = self.posterior_variance.gather(
                    -1,
                    timesteps
                ).reshape(
                    timesteps.shape[0], *((1,) * (len(z.shape) - 1))
                )

                posterior_mu = posterior_mu
                posterior_var = posterior_var.clamp(min=1e-20)

                z = posterior_mu + torch.sqrt(posterior_var) \
                    * (torch.randn_like(posterior_mu) if t > 0 else 0)
        self.unet.train()
        images = z / 2 + 0.5
        return images
