from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import global_network_dataset
from models import networks
from models.didn import DIDN
from util.metrics import psnr_per_sample, rmse_per_sample, ssim_per_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train vanilla MoDL or fine-tune RODIO from supplied DIDN weights."
    )
    parser.add_argument("--data-root", default=str(global_network_dataset.DEFAULT_DATA_ROOT))
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--name", default="modl_smug_4x")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Not specified by the paper; record this reproduction choice.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Not specified by the paper; record this reproduction choice.",
    )
    parser.add_argument("--lr-decay-start", type=int, default=10)
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--block-iter", type=int, default=6)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-weights",
        default=None,
        help="Pretrained vanilla MoDL/DIDN weights used to initialize RODIO fine-tuning.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Backward-compatible alias for --init-weights; does not restore optimizer state.",
    )
    parser.add_argument("--purified-dir", default=None)
    parser.add_argument("--val-purified-dir", default=None)
    parser.add_argument("--allow-untracked-purified", action="store_true")
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--expected-sigma-ft", type=float, default=0.01)
    parser.add_argument("--expected-val-noise-std", type=float, default=0.01)
    return parser.parse_args()


def device_and_ids(gpu_ids: str):
    ids = [int(x) for x in gpu_ids.split(",") if x.strip()]
    if ids and ids[0] >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{ids[0]}"), ids
    return torch.device("cpu"), []


def build_model(device: torch.device, gpu_ids: list[int]) -> nn.Module:
    net = DIDN(
        2,
        2,
        num_chans=64,
        pad_data=True,
        global_residual=True,
        n_res_blocks=2,
    ).float()
    networks.init_weights(net, init_type="normal", gain=0.02)
    if len(gpu_ids) > 1:
        net = nn.DataParallel(net, device_ids=gpu_ids)
    return net.to(device)


def state_dict_from_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced
        ckpt = torch.load(path, map_location="cpu")
    if hasattr(ckpt, "state_dict") and not isinstance(ckpt, dict):
        ckpt = ckpt.state_dict()
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return {k.removeprefix("module."): v for k, v in ckpt.items()}


def load_model_weights(net: nn.Module, path: str) -> None:
    target = net.module if isinstance(net, nn.DataParallel) else net
    target.load_state_dict(state_dict_from_checkpoint(path), strict=True)


def cg(output, tol, lambda_reg, smap, mask, data_image):
    return networks.CG.apply(
        output, tol, lambda_reg, smap, mask, data_image
    )


def recon(
    net: nn.Module,
    input_image,
    smap,
    mask,
    block_iter: int,
    cg_tol: float,
    lambda_reg: float,
):
    """Algorithm 3: purified input is both x0 and the DC data term."""
    output = input_image
    for _ in range(block_iter):
        denoised = net(output)
        output = cg(
            denoised,
            tol=cg_tol,
            lambda_reg=lambda_reg,
            smap=smap,
            mask=mask,
            data_image=input_image,
        )
    return output


def run_epoch(loader, net, optimizer, args, device, train: bool):
    net.train(train)
    totals = {"loss": 0.0, "rmse": 0.0, "psnr": 0.0, "ssim": 0.0, "n": 0}
    loss_fn = nn.MSELoss().to(device)
    iterator = tqdm(loader, leave=False, desc="train" if train else "val")
    for batch in iterator:
        input_image, target, smap, mask = [
            x.to(device).float() for x in batch[:4]
        ]
        with torch.set_grad_enabled(train):
            output = recon(
                net,
                input_image,
                smap,
                mask,
                args.block_iter,
                args.cg_tol,
                args.lambda_reg,
            )
            loss = loss_fn(output, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        rmse = rmse_per_sample(target.detach(), output.detach())
        psnr = psnr_per_sample(target.detach(), output.detach())
        ssim = ssim_per_sample(target.detach(), output.detach())
        bs = input_image.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * bs
        totals["rmse"] += float(rmse.sum().cpu())
        totals["psnr"] += float(psnr.sum().cpu())
        totals["ssim"] += float(ssim.sum().cpu())
        totals["n"] += bs

    n = totals.pop("n")
    if n == 0:
        return {}
    return {key: value / n for key, value in totals.items()}


def checkpoint_state(net, optimizer, scheduler, epoch, best_val, args):
    target = net.module if isinstance(net, nn.DataParallel) else net
    return {
        "model": target.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_rmse": best_val,
        "args": vars(args),
    }


def main() -> None:
    args = parse_args()
    if args.init_weights and args.resume:
        raise ValueError("Use only one of --init-weights and --resume")
    init_weights = args.init_weights or args.resume
    if args.purified_dir and not init_weights:
        raise ValueError(
            "RODIO fine-tuning must start from pretrained MoDL; pass --init-weights"
        )
    if args.purified_dir and not args.val_purified_dir:
        print(
            "WARNING: --val-purified-dir is absent; best checkpoint will be "
            "selected on clean rather than matched purified validation inputs."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device, gpu_ids = device_and_ids(args.gpu_ids)

    expr_dir = Path(args.checkpoints_dir) / args.name
    tracked_outputs = ("args.json", "history.json", "latest.pth", "vali_best.pth")
    if any((expr_dir / name).exists() for name in tracked_outputs):
        raise FileExistsError(
            f"Training run directory already contains tracked outputs: {expr_dir}. "
            "Use a new --name; optimizer-state resume is intentionally not implied."
        )
    expr_dir.mkdir(parents=True, exist_ok=True)
    with open(expr_dir / "args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    train_files, val_files, _ = global_network_dataset.split_files(
        args.data_root, args.train_size, args.val_size, args.test_size, args.seed
    )
    if args.purified_dir:
        train_manifest = global_network_dataset.require_matching_manifest(
            args.purified_dir, train_files, "training purification",
            args.allow_untracked_purified,
        )
        if train_manifest is not None:
            global_network_dataset.require_manifest_settings(
                train_manifest,
                {
                    "split": "train", "train_size": args.train_size,
                    "val_size": args.val_size, "test_size": args.test_size,
                    "acceleration": args.acceleration,
                    "sigma_ft": args.expected_sigma_ft,
                    "shift_fraction": 0.0, "measurement_dir": None,
                    "seed": args.seed,
                },
                "training purification",
            )
    if args.val_purified_dir:
        val_manifest = global_network_dataset.require_matching_manifest(
            args.val_purified_dir, val_files, "validation purification",
            args.allow_untracked_purified,
        )
        if val_manifest is not None:
            global_network_dataset.require_manifest_settings(
                val_manifest,
                {
                    "split": "val", "train_size": args.train_size,
                    "val_size": args.val_size, "test_size": args.test_size,
                    "acceleration": args.acceleration,
                    "noise_std": args.expected_val_noise_std,
                    "shift_fraction": 0.0, "measurement_dir": None,
                    "seed": args.seed,
                },
                "validation purification",
            )

    train_loader, val_loader, _ = global_network_dataset.build_loaders(
        data_root=args.data_root,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        acceleration=args.acceleration,
        seed=args.seed,
        num_workers=args.num_workers,
        purified_dir=args.purified_dir,
        val_purified_dir=args.val_purified_dir,
        noise_std=args.input_noise_std,
    )

    net = build_model(device, gpu_ids)
    if init_weights:
        load_model_weights(net, init_weights)
        print(f"Loaded strict initialization weights from {init_weights}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def lr_lambda(epoch: int):
        if epoch < args.lr_decay_start:
            return 1.0
        denominator = max(1, args.epochs - args.lr_decay_start)
        return max(0.0, 1.0 - (epoch - args.lr_decay_start + 1) / denominator)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    history = []
    best_val = float("inf")

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            train_loader, net, optimizer, args, device, train=True
        )
        val_metrics = run_epoch(
            val_loader, net, optimizer, args, device, train=False
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        with open(expr_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        current_val = val_metrics.get("rmse", float("inf"))
        if current_val < best_val:
            best_val = current_val
            torch.save(
                checkpoint_state(
                    net, optimizer, scheduler, epoch, best_val, args
                ),
                expr_dir / "vali_best.pth",
            )
        torch.save(
            checkpoint_state(net, optimizer, scheduler, epoch, best_val, args),
            expr_dir / "latest.pth",
        )

        print(
            f"epoch {epoch:03d} lr {optimizer.param_groups[0]['lr']:.2e} | "
            f"train rmse {train_metrics['rmse']:.5f} "
            f"psnr {train_metrics['psnr']:.3f} ssim {train_metrics['ssim']:.4f} | "
            f"val rmse {val_metrics['rmse']:.5f} "
            f"psnr {val_metrics['psnr']:.3f} ssim {val_metrics['ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
