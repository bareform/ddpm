from models import DDPM

import torch

import torchutils

@torch.no_grad()
def gen(ddpm: DDPM, z: torch.Tensor, dtype: str, device: torch.device) -> torch.Tensor:
    dtype = torchutils.get_torch_dtype(dtype)
    use_amp = (device.type == "cuda") and (dtype != torch.float32)
    ddpm.eval()
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
        batch_size = z.shape[0]
        z = torch.randn(batch_size, *ddpm.img_size, device=device, dtype=dtype)
        for t in range(ddpm.num_timesteps - 1, -1, -1):
            timesteps = torch.full(
                (z.shape[0],),
                t,
                dtype=torch.long,
                device=device,
            )
            _, x0 = ddpm(z, timesteps)
            z = z.to(dtype)
            x0 = x0.to(dtype)
            posterior_mu = (
                ddpm.posterior_mean_coefficient1.gather(0, timesteps)
                .reshape(z.shape[0], *((1,) * (len(z.shape) - 1)))
                * x0
            )
            posterior_mu = posterior_mu + (
                ddpm.posterior_mean_coefficient2.gather(0, timesteps)
                .reshape(z.shape[0], *((1,) * (len(z.shape) - 1)))
                * z
            )
            posterior_var = (
                ddpm.posterior_variance.gather(0, timesteps)
                .reshape(z.shape[0], *((1,) * (len(z.shape) - 1)))
            )
            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            z = posterior_mu + torch.sqrt(posterior_var.clamp(min=1e-20)) * noise
    images = (z + 1) / 2
    images = images * 255
    ddpm.train()
    return images
