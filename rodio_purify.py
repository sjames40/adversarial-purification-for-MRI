from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import global_network_dataset
from models import ncsnpp  # registers NCSN++ with models.utils
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from sampling import (
    LangevinCorrector,
    ReverseDiffusionPredictor,
    shared_corrector_update_fn,
    shared_predictor_update_fn,
)
from sde_lib import VESDE
from utils import fft2_m, get_data_inverse_scaler, get_data_scaler, ifft2_m, restore_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper-aligned RODIO Algorithm-2 purified inputs."
    )
    parser.add_argument("--data-root", default=str(global_network_dataset.DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--score-checkpoint",
        required=True,
        help="Pretrained score-MRI checkpoint, e.g. weights/checkpoint_95.pth.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--acceleration", type=float, default=4.0)
    parser.add_argument("--pst-step", type=int, default=150)
    parser.add_argument("--num-scales", type=int, default=500)
    parser.add_argument(
        "--sigma-ft",
        type=float,
        default=0.01,
        help="Per-real/imag-component k-space Gaussian standard deviation for training.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Per-real/imag-component k-space Gaussian standard deviation for val/test.",
    )
    parser.add_argument("--shift-fraction", type=float, default=0.0)
    parser.add_argument("--measurement-dir", default=None)
    parser.add_argument("--snr", type=float, default=0.16)
    parser.add_argument("--corrector-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def complex_from_ri(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1])


def ri_from_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((x.real, x.imag), dim=1).float()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_seed(base_seed: int, source: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{source}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def load_score_model(config, checkpoint: str, device: torch.device):
    config.device = device
    score_model = mutils.create_model(config).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(checkpoint, state, device, skip_sigma=True)
    ema.copy_to(score_model.parameters())
    score_model.eval()
    return score_model, int(state["step"])


def sense_data_consistency(
    image: torch.Tensor,
    measured_kspace: torch.Tensor,
    sensitivity: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply z <- z + A^H(y-Az) for A=MFS, exactly as Algorithm 1."""
    predicted = fft2_m(sensitivity * image[:, None])
    residual = mask * (measured_kspace - predicted)
    correction = torch.sum(torch.conj(sensitivity) * ifft2_m(residual), dim=1)
    return image + correction


def _score_update_complex(
    image: torch.Tensor,
    t: torch.Tensor,
    score_model,
    update_fn,
) -> torch.Tensor:
    real, _ = update_fn(image.real[:, None], t, score_model)
    imag, _ = update_fn(image.imag[:, None], t, score_model)
    return torch.complex(real[:, 0], imag[:, 0])


def purify_sense_images(
    score_model,
    sde: VESDE,
    config,
    image_init: torch.Tensor,
    measured_kspace: torch.Tensor,
    sensitivity: torch.Tensor,
    mask: torch.Tensor,
    pst_step: int,
    snr: float,
    corrector_steps: int,
):
    """Paper Algorithms 1-2 for multi-coil SENSE measurements.

    The score-MRI checkpoint is a one-channel magnitude-image prior. Following
    its official complex inference convention, the same score model is applied
    independently to the real and imaginary image components.
    """
    if not 1 <= int(pst_step) < sde.N:
        raise ValueError(f"pst_step must be in [1, {sde.N - 1}], got {pst_step}")
    if corrector_steps < 0:
        raise ValueError("corrector_steps must be non-negative")

    device = image_init.device
    sigmas = sde.discrete_sigmas.to(device)
    forward_std = torch.sqrt(
        (sigmas[pst_step] ** 2 - sigmas[0] ** 2).clamp_min(0.0)
    )
    # score-MRI's complex inference treats real and imaginary components as
    # independent one-channel samples; keep that convention for the checkpoint.
    image = image_init + forward_std * torch.complex(
        torch.randn_like(image_init.real), torch.randn_like(image_init.real)
    )

    predictor_update = lambda x, t, model: shared_predictor_update_fn(
        x,
        t,
        sde=sde,
        model=model,
        predictor=ReverseDiffusionPredictor,
        probability_flow=False,
        continuous=config.training.continuous,
    )
    # Algorithm 1 requires DC after every corrector update. Configure a single
    # Langevin step here and repeat explicitly below.
    corrector_update = lambda x, t, model: shared_corrector_update_fn(
        x,
        t,
        sde=sde,
        model=model,
        corrector=LangevinCorrector,
        continuous=config.training.continuous,
        snr=snr,
        n_steps=1,
    )

    with torch.no_grad():
        for step_id in tqdm(
            range(int(pst_step), 0, -1), leave=False, desc="DP reverse"
        ):
            predictor_t = torch.full(
                (image.shape[0],),
                step_id / (sde.N - 1),
                device=device,
                dtype=image.real.dtype,
            )
            # Algorithm 1 predicts from discrete state ``step_id`` to
            # ``step_id - 1``. The subsequent corrector therefore evaluates
            # the score at the new state/time rather than reusing predictor_t.
            corrector_t = torch.full(
                (image.shape[0],),
                (step_id - 1) / (sde.N - 1),
                device=device,
                dtype=image.real.dtype,
            )
            image = _score_update_complex(
                image, predictor_t, score_model, predictor_update
            )
            image = sense_data_consistency(
                image, measured_kspace, sensitivity, mask
            )

            for _ in range(corrector_steps):
                image = _score_update_complex(
                    image, corrector_t, score_model, corrector_update
                )
                image = sense_data_consistency(
                    image, measured_kspace, sensitivity, mask
                )

    return image


SCIENTIFIC_ARG_KEYS = (
    "data_root", "score_checkpoint", "split", "train_size", "val_size",
    "test_size", "acceleration", "pst_step", "num_scales", "sigma_ft",
    "noise_std", "shift_fraction", "measurement_dir", "snr",
    "corrector_steps", "seed", "limit",
)


def scientific_args(args) -> dict:
    return {key: getattr(args, key) for key in SCIENTIFIC_ARG_KEYS}


def build_manifest(args, files, score_step: int) -> dict:
    return {
        "schema_version": 3,
        "algorithm": "RODIO Algorithms 1-2; image-domain SENSE DC after predictor and each corrector",
        "score_model_complex_convention": "same one-channel score applied independently to real and imaginary components",
        "args": vars(args),
        "scientific_args": scientific_args(args),
        "score_checkpoint_sha256": sha256_file(args.score_checkpoint),
        "score_checkpoint_step": score_step,
        "sources": [p.name for p in files],
        "noise_parameterization": "noise_std and sigma_ft are per-real/imag-component standard deviations",
    }


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1 for deterministic per-source purification")
    if args.skip_existing and args.overwrite:
        raise ValueError("--skip-existing and --overwrite are mutually exclusive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files, val_files, test_files = global_network_dataset.split_files(
        args.data_root,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    files = {"train": train_files, "val": val_files, "test": test_files}[args.split]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        files = files[: args.limit]

    if args.measurement_dir:
        global_network_dataset.require_matching_manifest(
            args.measurement_dir, files, f"{args.split} measurement attack"
        )
    noise_std = args.sigma_ft if args.split == "train" else args.noise_std
    ds = global_network_dataset.SMUGKspaceDataset(
        files,
        acceleration=args.acceleration,
        seed=args.seed if args.split == "train" else args.seed + (10000 if args.split == "val" else 20000),
        noise_std=noise_std,
        shift_fraction=args.shift_fraction,
        measurement_dir=args.measurement_dir,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    configs = importlib.import_module(
        "configs.ve.fastmri_knee_320_ncsnpp_continuous"
    )
    config = configs.get_config()
    config.device = device
    config.training.batch_size = 1
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)
    if config.data.centered:
        raise ValueError(
            "The supplied checkpoint configuration is expected to use centered=False"
        )
    sde = VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        N=args.num_scales,
    )
    score_model, score_step = load_score_model(
        config, args.score_checkpoint, device
    )

    manifest = build_manifest(args, files, score_step)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        if not (args.overwrite or args.skip_existing):
            raise FileExistsError(
                f"{manifest_path} already exists; use a new output directory, "
                "--skip-existing, or --overwrite"
            )
        if args.skip_existing:
            with open(manifest_path, encoding="utf-8") as handle:
                existing = json.load(handle)
            comparable_keys = (
                "algorithm", "score_model_complex_convention",
                "scientific_args", "score_checkpoint_sha256",
                "score_checkpoint_step", "sources",
            )
            mismatch = [
                key for key in comparable_keys
                if existing.get(key) != manifest.get(key)
            ]
            if mismatch:
                raise ValueError(
                    "Refusing to mix existing purified files with different "
                    f"provenance; mismatched manifest fields: {mismatch}. "
                    "Use a new output directory."
                )
        else:
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
    else:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    for batch in tqdm(loader, desc=f"purify {args.split}"):
        input_image, _target, smap, _mask_ri, coil_mask, _clean_kspace, measured_ri, fnames = batch
        fname = fnames[0]
        output_path = out_dir / f"{Path(fname).stem}.npz"
        if output_path.exists():
            if args.skip_existing:
                continue
            if not args.overwrite:
                raise FileExistsError(
                    f"{output_path} exists; use --overwrite or --skip-existing"
                )

        sample_seed = source_seed(args.seed, fname)
        torch.manual_seed(sample_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(sample_seed)

        input_image = input_image.to(device).float()
        smap = smap.to(device).float()
        coil_mask = coil_mask.to(device).float()
        measured_ri = measured_ri.to(device).float()

        image_init = complex_from_ri(input_image)
        sensitivity = torch.complex(smap[:, :, 0], smap[:, :, 1])
        measured = torch.complex(measured_ri[:, :, 0], measured_ri[:, :, 1])
        measured = measured * coil_mask

        purified = purify_sense_images(
            score_model=score_model,
            sde=sde,
            config=config,
            image_init=scaler(image_init),
            measured_kspace=measured,
            sensitivity=sensitivity,
            mask=coil_mask,
            pst_step=args.pst_step,
            snr=args.snr,
            corrector_steps=args.corrector_steps,
        )
        purified = inverse_scaler(purified)
        x0 = ri_from_complex(purified).detach().cpu().numpy()[0]
        if x0.shape != (2, 320, 320) or not np.isfinite(x0).all():
            raise RuntimeError(f"Invalid purified output for {fname}: {x0.shape}")
        np.savez_compressed(
            output_path,
            x0=x0.astype(np.float32),
            source=fname,
            source_seed=np.int64(sample_seed),
        )


if __name__ == "__main__":
    main()
