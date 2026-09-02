from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import global_network_dataset
from models.didn import DIDN
from train_MoDL import load_model_weights, recon
from util.metrics import psnr_per_sample, rmse_per_sample, ssim_per_sample


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate vanilla MoDL, DP+MoDL, or fine-tuned RODIO."
    )
    p.add_argument("--data-root", default=str(global_network_dataset.DEFAULT_DATA_ROOT))
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--checkpoint-kind",
        choices=["vanilla", "rodio_finetuned"],
        default="vanilla",
        help="Labels the scientific method correctly; it does not change the architecture.",
    )
    p.add_argument("--output-json", default="eval_results.json")
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=20)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--acceleration", type=float, default=4.0)
    p.add_argument("--shift-fraction", type=float, default=0.0)
    p.add_argument("--measurement-dir", default=None)
    p.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Per-real/imag-component complex k-space Gaussian standard deviation.",
    )
    p.add_argument(
        "--purified-dir",
        default=None,
        help="Matched Algorithm-2 purified x0 directory.",
    )
    p.add_argument("--allow-untracked-purified", action="store_true")
    p.add_argument("--block-iter", type=int, default=6)
    p.add_argument("--lambda-reg", type=float, default=1.0)
    p.add_argument("--cg-tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def method_name(args) -> str:
    if args.purified_dir and args.checkpoint_kind == "rodio_finetuned":
        return "RODIO (Algorithm 2 + fine-tuned MoDL)"
    if args.purified_dir:
        return "DP + pretrained vanilla MoDL (paper ablation, not full RODIO)"
    if args.checkpoint_kind == "rodio_finetuned":
        return "fine-tuned MoDL without test-time DP"
    return "vanilla MoDL"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    _, _, test_files = global_network_dataset.split_files(
        args.data_root,
        args.train_size,
        args.val_size,
        args.test_size,
        args.seed,
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        test_files = test_files[:args.limit]
    if args.purified_dir:
        purification_manifest = global_network_dataset.require_matching_manifest(
            args.purified_dir, test_files, "test purification",
            args.allow_untracked_purified,
        )
        if purification_manifest is not None:
            global_network_dataset.require_manifest_settings(
                purification_manifest,
                {
                    "split": "test",
                    "train_size": args.train_size,
                    "val_size": args.val_size,
                    "test_size": args.test_size,
                    "acceleration": args.acceleration,
                    "noise_std": args.noise_std,
                    "shift_fraction": args.shift_fraction,
                    "measurement_dir": args.measurement_dir,
                    "seed": args.seed,
                },
                "test purification",
            )
    if args.measurement_dir:
        global_network_dataset.require_matching_manifest(
            args.measurement_dir, test_files, "test measurement attack"
        )

    ds = global_network_dataset.SMUGKspaceDataset(
        test_files,
        acceleration=args.acceleration,
        seed=args.seed + 20000,
        noise_std=args.noise_std,
        purified_dir=args.purified_dir,
        shift_fraction=args.shift_fraction,
        measurement_dir=args.measurement_dir,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    net = DIDN(
        2,
        2,
        num_chans=64,
        pad_data=True,
        global_residual=True,
        n_res_blocks=2,
    ).float().to(device)
    load_model_weights(net, args.checkpoint)
    net.eval()

    rows = []
    for batch in tqdm(loader, desc="eval"):
        input_image, target, smap, mask = [
            x.to(device).float() for x in batch[:4]
        ]
        fnames = list(batch[-1])
        with torch.no_grad():
            output = recon(
                net,
                input_image,
                smap,
                mask,
                args.block_iter,
                args.cg_tol,
                args.lambda_reg,
            )
            rmse = rmse_per_sample(target, output).detach().cpu().numpy()
            psnr = psnr_per_sample(target, output).detach().cpu().numpy()
            ssim = ssim_per_sample(target, output).detach().cpu().numpy()

        for i, fname in enumerate(fnames):
            rows.append(
                {
                    "file": fname,
                    "rmse": float(rmse[i]),
                    "psnr": float(psnr[i]),
                    "ssim": float(ssim[i]),
                }
            )

    if len(rows) != len(test_files):
        raise RuntimeError(f"Expected {len(test_files)} results, got {len(rows)}")
    result = {
        "method": method_name(args),
        "args": vars(args),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "summary": {
            metric: summarize([row[metric] for row in rows])
            for metric in ("rmse", "psnr", "ssim")
        },
        "per_image": rows,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"method": result["method"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
