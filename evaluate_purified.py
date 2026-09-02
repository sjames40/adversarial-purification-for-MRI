from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import global_network_dataset
from util.metrics import psnr_per_sample, rmse_per_sample, ssim_per_sample


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate standalone Algorithm-2 DP outputs.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--purified-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=20)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--acceleration", type=float, default=4.0)
    p.add_argument("--shift-fraction", type=float, default=0.0)
    p.add_argument("--measurement-dir", default=None)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "std": float(array.std(ddof=0)),
        "median": float(np.median(array)), "min": float(array.min()),
        "max": float(array.max()),
    }


def main():
    args = parse_args()
    _, _, test_files = global_network_dataset.split_files(
        args.data_root, args.train_size, args.val_size, args.test_size, args.seed
    )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        test_files = test_files[:args.limit]
    purification_manifest = global_network_dataset.require_matching_manifest(
        args.purified_dir, test_files, "standalone-DP test purification"
    )
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
        "standalone-DP test purification",
    )
    if args.measurement_dir:
        global_network_dataset.require_matching_manifest(
            args.measurement_dir, test_files, "standalone-DP test measurement"
        )
    dataset = global_network_dataset.SMUGKspaceDataset(
        test_files, acceleration=args.acceleration, seed=args.seed + 20000,
        noise_std=args.noise_std, purified_dir=args.purified_dir,
        shift_fraction=args.shift_fraction, measurement_dir=args.measurement_dir,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(args.device)
    rows = []
    for batch in tqdm(loader, desc="eval standalone DP"):
        purified, target = [x.to(device).float() for x in batch[:2]]
        names = list(batch[-1])
        rmse = rmse_per_sample(target, purified).cpu().numpy()
        psnr = psnr_per_sample(target, purified).cpu().numpy()
        ssim = ssim_per_sample(target, purified).cpu().numpy()
        for i, name in enumerate(names):
            rows.append({
                "file": name, "rmse": float(rmse[i]),
                "psnr": float(psnr[i]), "ssim": float(ssim[i]),
            })
    if len(rows) != len(test_files):
        raise RuntimeError(f"Expected {len(test_files)} results, got {len(rows)}")
    result = {
        "method": "standalone DP (Algorithm 2, no MoDL)",
        "args": vars(args),
        "summary": {
            key: summarize([row[key] for row in rows])
            for key in ("rmse", "psnr", "ssim")
        },
        "per_image": rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"method": result["method"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
