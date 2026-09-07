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
from attacks import measurement_attack
from models.didn import DIDN
from train_MoDL import load_model_weights


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODL_CHECKPOINT = (
    PROJECT_ROOT / "weights" / "DIDN_lambda1_3000_images_trained.pt"
)
DEFAULT_ATTACK_DIR = PROJECT_ROOT / "runs" / "attack_test_pgd_eps0004"


def parse_args():
    p = argparse.ArgumentParser(description="Generate paper-style measurement-space attacks.")
    p.add_argument("--data-root", default=str(global_network_dataset.DEFAULT_DATA_ROOT))
    p.add_argument("--checkpoint", default=str(DEFAULT_MODL_CHECKPOINT))
    p.add_argument("--output-dir", default=str(DEFAULT_ATTACK_DIR))
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=20)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--acceleration", type=float, default=4.0)
    p.add_argument("--method", choices=["pgd", "momentum"], default="pgd")
    p.add_argument(
        "--reference",
        choices=["clean_reconstruction", "ground_truth"],
        default="clean_reconstruction",
    )
    p.add_argument(
        "--loss-domain",
        choices=["complex", "magnitude"],
        default="complex",
        help="Released code uses two-channel complex MSE; paper leaves L generic.",
    )
    p.add_argument("--epsilon", type=float, default=0.004)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--step-size", type=float, default=None)
    p.add_argument("--no-random-start", action="store_true")
    p.add_argument("--block-iter", type=int, default=6)
    p.add_argument("--lambda-reg", type=float, default=1.0)
    p.add_argument("--cg-tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.method == "momentum":
        print(
            "WARNING: momentum mode is a documented MI-FGSM-style approximation; "
            "the paper does not publish enough AUTO optimizer details for an exact reproduction."
        )
    step_size = args.step_size if args.step_size is not None else args.epsilon / 3.0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    train, val, test = global_network_dataset.split_files(
        args.data_root, args.train_size, args.val_size, args.test_size, args.seed
    )
    files = {"train": train, "val": val, "test": test}[args.split]
    if args.limit is not None:
        files = files[: args.limit]
    split_seed = args.seed if args.split == "train" else args.seed + (10000 if args.split == "val" else 20000)
    dataset = global_network_dataset.SMUGKspaceDataset(
        files, acceleration=args.acceleration, seed=split_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DIDN(
        2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=2
    ).float().to(device)
    load_model_weights(model, args.checkpoint)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} exists; use a new output directory or --overwrite"
        )
    manifest = {
        "schema_version": 1,
        "args": vars(args),
        "effective_step_size": step_size,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "sources": [p.name for p in files],
        "constraint": "per-real/imag-component L_inf on acquired k-space only",
        "auto_note": (
            None
            if args.method == "pgd"
            else "MI-FGSM-style momentum approximation; not exact unpublished AUTO"
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    for batch in tqdm(loader, desc=f"{args.method} {args.split}"):
        _input, target, smap, mask, _coil_mask, clean_kspace = batch[:6]
        fname = batch[-1][0]
        output_path = output_dir / f"{Path(fname).stem}.npz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite")

        target = target.to(device).float()
        smap = smap.to(device).float()
        mask = mask.to(device).float()
        clean_kspace = clean_kspace.to(device).float()
        adversarial, delta = measurement_attack(
            model=model,
            clean_kspace=clean_kspace,
            target=target,
            sensitivity=smap,
            mask=mask,
            epsilon=args.epsilon,
            step_size=step_size,
            steps=args.steps,
            method=args.method,
            reference=args.reference,
            block_iter=args.block_iter,
            cg_tol=args.cg_tol,
            lambda_reg=args.lambda_reg,
            random_start=not args.no_random_start,
            loss_domain=args.loss_domain,
        )
        adversarial_np = adversarial[0].cpu().numpy().astype(np.float32)
        delta_np = delta[0].cpu().numpy().astype(np.float32)
        np.savez_compressed(
            output_path,
            measured_kspace=adversarial_np,
            delta=delta_np,
            source=fname,
        )


if __name__ == "__main__":
    main()
