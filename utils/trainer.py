from models import DDPM

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from cleanfid import fid
from datasets import load_dataset
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import torchutils

from .ema_utils import (
    EMA
)
from .gen_utils import (
    gen
)
from .hf_utils import (
    ImageDataset
)

def parse_args():
    parser = torchutils.ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pretrained training configuration.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["afhq"],
        help="Dataset to train on. Must be one of: `afhq`.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads to use in data loading (default: 4)",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Whether to pin memory for data loading (default: True)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=200000,
        help="Number of iterations to train for (default: 200000)",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=64,
        help="Height of the input images (default: 64)",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=64,
        help="Width of the input images (default: 64)",
    )
    parser.add_argument(
        "--ddpm_in_channels",
        type=int,
        nargs="+",
        help="Input channels for the DDPM.",
    )
    parser.add_argument(
        "--ddpm_timesteps",
        type=int,
        default=1000,
        help="Number of denoising steps (default: 1000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001).",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 (default: 0.9).",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--ema_weight_decay",
        type=float,
        default=0.999,
        help="EMA weight decay (default: 0.999).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(".", "checkpoints"),
        help="Directory to save checkpoints (default: ./checkpoints).",
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=500000,
        help="Interval (in iterations) at which to save model checkpoints (default: 500000).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(".", "output"),
        help="Directory to save intermediate output (default: ./output).",
    )
    parser.add_argument(
        "--save_output_interval",
        type=int,
        default=100000,
        help="Number of iterations between saving intermediate output (default: 100000).",
    )
    parser.add_argument(
        "--nrow",
        type=int,
        default=9,
        help="Number of rows in output grid (default: 9).",
    )
    parser.add_argument(
        "--compute_fid_interval",
        type=int,
        default=100000,
        help="Number of iterations between FID evaluations (default: 100000).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Train with torch.compile.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    args = parser.parse_args()    
    return args 

def main():
    args = parse_args()
    pad_length = len(str(args.n_steps))

    torchutils.set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Training on {args.dataset}")
    if args.dataset == "afhq":
        hf_dataset = load_dataset("luethan2025/AFHQ-64x64", split="train")
        transform = transforms.Compose([
            transforms.Resize(args.image_height),
            transforms.CenterCrop(args.image_height),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = ImageDataset(
            hf_dataset=hf_dataset,
            transform=transform
        )
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )
        def cycle(dataloader):
            while True:
                for batch in dataloader:
                    yield batch
        dataloader = cycle(dataloader)
        dataset = "afhq"
        img_size = (3, args.image_height, args.image_width)

    data_dir = os.path.join("ddpm_data", dataset)
    os.makedirs(data_dir, exist_ok=True)
    file_pad = len(str(len(hf_dataset)))
    for idx, item in enumerate(hf_dataset):
        image = item['image']
        image.save(os.path.join(data_dir, f"{str(idx).zfill(file_pad)}.png"))

    if not fid.test_stats_exists(dataset, mode="clean"):
        fid.make_custom_stats(dataset, data_dir, mode="clean")

    ddpm = DDPM(
        img_size=img_size,
        in_channels=args.ddpm_in_channels,
        timesteps=args.ddpm_timesteps
    )
    ddpm = ddpm.to(device)
    ddpm.train()
    ema = EMA(ddpm, decay=args.ema_weight_decay)

    optimizer = optim.Adam(ddpm.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    criterion = nn.MSELoss()

    dtype = torchutils.get_torch_dtype(args.mixed_precision)
    use_amp = (device.type == "cuda") and (dtype != torch.float32)
    use_scaler = (device.type == "cuda") and (dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    if args.use_torch_compile:
        print("Compiling")
        ddpm.unet = torch.compile(ddpm.unet, mode="default")

        warmup_images = torch.randn(
            args.batch_size,
            *img_size,
            device=device
        )

        warmup_time = torch.randint(
            0,
            ddpm.num_timesteps,
            (args.batch_size,),
            device=device,
            dtype=torch.long,
        )

        warmup_noise = torch.randn_like(warmup_images)

        warmup_images = warmup_images
        warmup_noise = warmup_noise

        alpha_bar = ddpm.alphas_cumprod.gather(
            0, warmup_time
        ).reshape(
            warmup_time.shape[0],
            *((1,) * (warmup_images.ndim - 1))
        )

        warmup_xt = (
            torch.sqrt(alpha_bar) * warmup_images
            + torch.sqrt(1. - alpha_bar) * warmup_noise
        )

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            predicted_noise, _ = ddpm(warmup_xt, warmup_time)
            warmup_loss = criterion(predicted_noise, warmup_noise)

        warmup_loss.backward()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        print("Finished warmup")

    if not (os.path.exists(args.checkpoint_dir) and os.path.isdir(args.checkpoint_dir)):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    output_dir = os.path.join(args.output_dir, args.dataset, "epoch")
    if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)

    global_step = 0
    pbar = tqdm(range(args.n_steps))

    for _ in pbar:
        images = next(dataloader)
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]

        time = torch.randint(
            0,
            ddpm.num_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(images)

        alpha_bar = ddpm.alphas_cumprod.gather(
            0, time
        ).reshape(
            time.shape[0],
            *((1,) * (images.ndim - 1))
        )

        xt = (
            torch.sqrt(alpha_bar) * images
            + torch.sqrt(1. - alpha_bar) * noise
        )

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            predicted_noise, _ = ddpm(xt, time)
            loss = criterion(predicted_noise, noise)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        
        ema.update(ddpm)

        if (global_step + 1) % args.save_output_interval == 0:
            print("Saving fake images")
            ddpm.eval()
            with torch.no_grad():
                fake_images = ddpm.generate_samples(num_samples=args.nrow*args.nrow, dtype=dtype)
                grid = torchvision.utils.make_grid(
                    fake_images,
                    nrow=args.nrow,
                    normalize=True,
                    value_range=(-1, 1)
                )
                torchvision.utils.save_image(
                    grid,
                    os.path.join(output_dir, f"{global_step + 1:0{pad_length}d}.png")
                )
            ddpm.train()

        if (global_step + 1) % args.save_checkpoint_interval == 0:
            print("Saving model checkpoints")
            state_dict = OrderedDict()
            for k, v in ema.ema_model.state_dict().items():
                new_key = k.replace("_orig_mod.", "")
                state_dict[new_key] = v
            checkpoint_file = os.path.join(args.checkpoint_dir, f"{args.dataset}_checkpoint_{global_step+ 1:0{pad_length}d}.pth")
            torch.save({
                "dataset": args.dataset,
                "ddpm": state_dict,
                "in_channels": args.ddpm_in_channels,
                "timesteps": args.ddpm_timesteps,
                "lr": args.lr,
                "optimizer": optimizer.state_dict(),
                "image_height": args.image_height,
                "image_width": args.image_width
            }, checkpoint_file)
            checkpoint_file = os.path.join(args.checkpoint_dir, f"{args.dataset}_{global_step+ 1:0{pad_length}d}.pth")
            torch.save({
                "dataset": args.dataset,
                "ddpm": state_dict,
                "img_size": img_size,
                "in_channels": args.ddpm_in_channels,
                "timesteps": args.ddpm_timesteps,
            }, checkpoint_file)
            
        if (global_step + 1) % args.compute_fid_interval == 0:
            fid_score = fid.compute_fid(
                gen=lambda z: gen(ddpm, z, dtype, device),
                dataset_name="afhq",
                dataset_res=64,
                num_gen=10_000,
                dataset_split="custom"
            )
            print(fid_score)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

        global_step += 1

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    state_dict = OrderedDict()
    for k, v in ema.ema_model.state_dict().items():
        new_key = k.replace("_orig_mod.", "")
        state_dict[new_key] = v

    checkpoint_file = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.n_steps}.pth")
    torch.save({
        "dataset": args.dataset,
        "ddpm": state_dict,
        "img_size": img_size,
        "in_channels": args.ddpm_in_channels,
        "timesteps": args.ddpm_timesteps,
    }, checkpoint_file)

if __name__ == "__main__":
    main()
