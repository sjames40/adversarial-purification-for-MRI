from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import global_network_dataset
from sde_lib import VESDE
from util.metrics import magnitude


def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate paper PST using Eq. (13) on clean and PGD-perturbed A^H y."
    )
    p.add_argument("--data-root", required=True)
    p.add_argument("--perturbed-measurement-dir", required=True,
                   help="Matched NPZ measurements from generate_kspace_attack.py.")
    p.add_argument("--output-json", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=20)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--num-samples", type=int, default=20,
                   help="Paper Fig. 3 uses 20 scans.")
    p.add_argument("--acceleration", type=float, default=4.0)
    p.add_argument("--num-scales", type=int, default=500)
    p.add_argument("--sigma-min", type=float, default=0.01)
    p.add_argument("--sigma-max", type=float, default=378.0)
    p.add_argument("--max-step", type=int, default=499)
    p.add_argument("--step-stride", type=int, default=1)
    p.add_argument("--relative-threshold", type=float, default=0.05,
                   help="Operational approximately-zero tolerance; unpublished in paper.")
    p.add_argument("--absolute-threshold", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def selected_files(args):
    train, val, test = global_network_dataset.split_files(
        args.data_root, args.train_size, args.val_size, args.test_size, args.seed
    )
    files = {"train": train, "val": val, "test": test}[args.split]
    if args.num_samples <= 1:
        raise ValueError("--num-samples must exceed one for unbiased MMD")
    if args.num_samples > len(files):
        raise ValueError(
            f"Requested {args.num_samples} samples from {args.split}, "
            f"but the split contains {len(files)}"
        )
    return files[:args.num_samples]


def collect_adjoint_images(args, files, measurement_dir=None):
    split_seed = args.seed if args.split == "train" else args.seed + (
        10000 if args.split == "val" else 20000
    )
    dataset = global_network_dataset.SMUGKspaceDataset(
        files, acceleration=args.acceleration, seed=split_seed,
        measurement_dir=measurement_dir,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    images, names = [], []
    for batch in tqdm(
        loader, desc="load clean" if measurement_dir is None else "load perturbed"
    ):
        images.append(batch[0].float())
        names.append(batch[-1][0])
    return torch.cat(images), names


def mmd_rbf_unbiased(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> float:
    """Eq. (13): unbiased within-set terms and all cross-set pairs."""
    if x.shape[0] != y.shape[0] or x.shape[0] <= 1:
        raise ValueError("MMD needs equal clean/perturbed counts greater than one")
    if bandwidth <= 0:
        raise ValueError(f"Kernel bandwidth must be positive, got {bandwidth}")
    x, y = x.flatten(1), y.flatten(1)
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    kxx = torch.exp(-gamma * torch.cdist(x, x).square())
    kyy = torch.exp(-gamma * torch.cdist(y, y).square())
    kxy = torch.exp(-gamma * torch.cdist(x, y).square())
    n = x.shape[0]
    within_x = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
    within_y = (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
    return float((within_x + within_y - 2.0 * kxy.mean()).cpu())


def main():
    args = parse_args()
    if not 0 <= args.max_step < args.num_scales:
        raise ValueError("--max-step must be in [0, num-scales-1]")
    if args.step_stride <= 0:
        raise ValueError("--step-stride must be positive")
    if not 0 <= args.relative_threshold <= 1:
        raise ValueError("--relative-threshold must be in [0, 1]")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    files = selected_files(args)
    global_network_dataset.require_matching_manifest(
        args.perturbed_measurement_dir, files, "PST measurement attack"
    )
    clean, clean_names = collect_adjoint_images(args, files)
    perturbed, perturbed_names = collect_adjoint_images(
        args, files, args.perturbed_measurement_dir
    )
    if clean_names != perturbed_names:
        raise RuntimeError("Clean and perturbed sources are not in identical order")
    clean, perturbed = clean.to(device), perturbed.to(device)

    # Paper Sec. V-C: kernel v is mean magnitude of clean set Z.
    bandwidth = float(magnitude(clean).mean().cpu())
    if bandwidth <= 0:
        raise RuntimeError("Mean clean-image magnitude is zero")

    sde = VESDE(args.sigma_min, args.sigma_max, args.num_scales)
    sigmas = sde.discrete_sigmas.to(device)
    clean_noise = torch.randn(
        clean.shape,
        generator=torch.Generator(device=device).manual_seed(args.seed + 12345),
        device=device,
    )
    perturbed_noise = torch.randn(
        perturbed.shape,
        generator=torch.Generator(device=device).manual_seed(args.seed + 54321),
        device=device,
    )

    rows, selected, initial_mmd = [], None, None
    for step in tqdm(range(0, args.max_step + 1, args.step_stride),
                     desc="Eq. (13) MMD"):
        std = torch.sqrt((sigmas[step].square() - sigmas[0].square()).clamp_min(0))
        value = mmd_rbf_unbiased(
            clean + std * clean_noise, perturbed + std * perturbed_noise, bandwidth
        )
        nonnegative = max(value, 0.0)
        if initial_mmd is None:
            initial_mmd = max(nonnegative, 1e-12)
        threshold = max(args.absolute_threshold, args.relative_threshold * initial_mmd)
        rows.append({
            "step": step, "mmd_unbiased": value,
            "mmd_clipped_for_threshold": nonnegative, "threshold": threshold,
        })
        if selected is None and nonnegative <= threshold:
            selected = step

    if selected is None:
        selected = rows[-1]["step"]
    result = {
        "schema_version": 2,
        "selected_pst_step": selected,
        "paper_reference_step": 150,
        "bandwidth_mean_clean_magnitude": bandwidth,
        "selection_note": (
            "The paper says MMD approximately zero but publishes no tolerance; "
            "selected_pst_step uses the explicit thresholds in args. "
            "Use PST=150 for the main paper reproduction."
        ),
        "sources": clean_names, "args": vars(args), "mmd": rows,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "selected_pst_step": selected, "paper_reference_step": 150,
        "bandwidth": bandwidth,
    }, indent=2))


if __name__ == "__main__":
    main()
