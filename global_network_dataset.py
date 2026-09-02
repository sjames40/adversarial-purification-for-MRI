"""SMUG/fastMRI-style multi-coil data loading for MoDL and RODIO.

The original repository contained hard-coded paths and returned only a test
loader. This module is intentionally self-contained: it reads the available
SMUG NPZ files with keys s_r, s_i, k_r, k_i, creates deterministic Cartesian
variable-density masks, and returns tensors in the layout expected by the
MoDL SENSE operators in models/networks.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models import networks
from util.util import complex_conj, complex_matmul, fft2, ifft2

DEFAULT_DATA_ROOT = Path("/egr/research-slim/shared/baselines/SMUG/SMUG_journal-main/data")
DEFAULT_KSPACE_DIR = DEFAULT_DATA_ROOT / "NEW_KSPACE"


def _resolve_kspace_dir(root: os.PathLike | str) -> Path:
    root = Path(root)
    if root.is_dir() and any(root.glob("*.npz")):
        return root
    candidate = root / "NEW_KSPACE"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Could not find NPZ k-space files under {root}")


def make_vdrs_mask(
    height: int,
    width: int,
    acceleration: float = 4.0,
    center_fraction: Optional[float] = None,
    seed: int = 0,
    shift_fraction: float = 0.0,
) -> np.ndarray:
    """Create a Cartesian variable-density random sampling mask.

    The paper uses 4x acceleration, i.e. roughly 25 percent phase-encode
    sampling, with a fully sampled low-frequency band. The default central
    fraction follows the original code comments: 0.32 / acceleration.
    """
    if acceleration <= 0:
        raise ValueError(f"acceleration must be positive, got {acceleration}")
    if not 0.0 <= shift_fraction < 1.0:
        raise ValueError(f"shift_fraction must be in [0, 1), got {shift_fraction}")
    if acceleration <= 1:
        return np.ones((height, width), dtype=bool)

    rng = np.random.default_rng(seed)
    center_fraction = 0.32 / acceleration if center_fraction is None else center_fraction
    num_total = max(1, int(round(width / acceleration)))
    num_center = max(1, int(round(width * center_fraction)))
    num_center = min(num_center, num_total)

    center_start = width // 2 - num_center // 2
    center_stop = center_start + num_center
    selected = set(range(center_start, center_stop))

    remaining = num_total - num_center
    if remaining > 0:
        candidates = np.array([i for i in range(width) if i not in selected])
        distances = np.abs(candidates - (width - 1) / 2.0)
        probs = 1.0 / (distances + 1.0)
        probs /= probs.sum()
        sampled = rng.choice(candidates, size=remaining, replace=False, p=probs)
        if shift_fraction:
            # Shift in the ordered set outside the ACS band. This is a
            # bijection, so the shift cannot change the acceleration factor.
            high_frequency = np.asarray(
                [i for i in range(width) if not center_start <= i < center_stop], dtype=np.int64
            )
            rank = {int(line): j for j, line in enumerate(high_frequency)}
            shift = int(round(width * shift_fraction))
            sampled = np.asarray(
                [high_frequency[(rank[int(line)] + shift) % len(high_frequency)] for line in sampled],
                dtype=np.int64,
            )
        selected.update(int(i) for i in sampled)

    mask = np.zeros((height, width), dtype=bool)
    mask[:, sorted(selected)] = True
    if int(mask[0].sum()) != num_total:
        raise RuntimeError(
            f"Mask construction changed the requested line count: expected {num_total}, "
            f"got {int(mask[0].sum())}"
        )
    if not np.all(mask[:, center_start:center_stop]):
        raise RuntimeError("Mask construction changed the fully sampled ACS band")
    return mask


def center_crop_last2(array: np.ndarray, crop_size: int = 320) -> np.ndarray:
    h, w = array.shape[-2:]
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot crop {array.shape} to {crop_size}x{crop_size}")
    top = h // 2 - crop_size // 2
    left = w // 2 - crop_size // 2
    return array[..., top : top + crop_size, left : left + crop_size]


def _normalise_full_sample(coil_images: torch.Tensor, smaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sense = torch.sum(complex_matmul(coil_images, complex_conj(smaps)), dim=0)
    scale = torch.max(torch.sqrt(torch.sum(sense * sense, dim=0))).clamp_min(1e-8)
    return coil_images / scale, smaps


class SMUGKspaceDataset(Dataset):
    """Dataset returning MoDL-ready SENSE tensors.

    Return tuple:
      input_image: [2, 320, 320] aliased A^H y or saved purified x0
      target: [2, 320, 320] fully sampled SENSE target
      smap: [15, 2, 320, 320] coil sensitivity maps
      mask: [2, 320, 320] image-space two-channel mask for CG operators
      coil_mask: [15, 320, 320] mask for complex multi-coil diffusion routines
      kspace: [15, 2, 320, 320] normalized clean full k-space
      measured_kspace: [15, 2, 320, 320] normalized measured k-space, possibly noisy
      fname: source NPZ basename
    """

    def __init__(
        self,
        files: Sequence[os.PathLike | str],
        acceleration: float = 4.0,
        crop_size: int = 320,
        seed: int = 0,
        noise_std: float = 0.0,
        purified_dir: Optional[os.PathLike | str] = None,
        shift_fraction: float = 0.0,
        measurement_dir: Optional[os.PathLike | str] = None,
    ) -> None:
        self.files = [Path(f) for f in files]
        self.acceleration = acceleration
        self.crop_size = crop_size
        self.seed = seed
        self.noise_std = noise_std
        self.purified_dir = Path(purified_dir) if purified_dir else None
        self.shift_fraction = shift_fraction
        self.measurement_dir = Path(measurement_dir) if measurement_dir else None
        if self.measurement_dir is not None and self.noise_std > 0:
            raise ValueError("measurement_dir and noise_std are mutually exclusive")

    def __len__(self) -> int:
        return len(self.files)

    def _load_npz(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with np.load(path) as data:
            required = {"s_r", "s_i", "k_r", "k_i"}
            missing = required.difference(data.files)
            if missing:
                raise KeyError(f"{path} is missing required arrays: {sorted(missing)}")
            s_r = data["s_r"].astype(np.float32) / 32767.0
            s_i = data["s_i"].astype(np.float32) / 32767.0
            k_r = data["k_r"].astype(np.float32) / 32767.0
            k_i = data["k_i"].astype(np.float32) / 32767.0

        if s_r.shape != s_i.shape or k_r.shape != k_i.shape:
            raise ValueError(f"Real/imaginary array shapes disagree in {path}")
        if s_r.ndim != 3 or k_r.ndim != 3 or s_r.shape[0] != k_r.shape[0]:
            raise ValueError(
                f"Expected [coils,H,W] sensitivity maps and k-space in {path}; "
                f"got smap={s_r.shape}, kspace={k_r.shape}"
            )
        if s_r.shape[0] != 15:
            raise ValueError(f"The paper setting requires 15 coils; {path} contains {s_r.shape[0]}")
        if not all(np.isfinite(x).all() for x in (s_r, s_i, k_r, k_i)):
            raise ValueError(f"Non-finite values found in {path}")

        smap_np = np.stack((center_crop_last2(s_r, self.crop_size), center_crop_last2(s_i, self.crop_size)), axis=1)
        k_np = np.stack((k_r, k_i), axis=1)

        k_full = torch.from_numpy(k_np)
        coil_images = ifft2(k_full.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        coil_images = center_crop_last2(coil_images, self.crop_size)
        smaps = torch.from_numpy(smap_np)
        coil_images, smaps = _normalise_full_sample(coil_images.float(), smaps.float())
        kspace = fft2(coil_images.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return kspace.float(), smaps.float(), coil_images.float()

    def _mask(self, index: int, num_coils: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_np = make_vdrs_mask(
            self.crop_size,
            self.crop_size,
            acceleration=self.acceleration,
            seed=self.seed + index,
            shift_fraction=self.shift_fraction,
        )
        mask = torch.from_numpy(np.repeat(mask_np[None, ...], 2, axis=0)).float()
        coil_mask = torch.from_numpy(np.repeat(mask_np[None, ...], num_coils, axis=0)).float()
        return mask, coil_mask

    def _maybe_load_purified(self, fname: str) -> Optional[torch.Tensor]:
        if self.purified_dir is None:
            return None
        path = self.purified_dir / f"{Path(fname).stem}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"purified_dir was specified but the matched input is missing: {path}"
            )
        with np.load(path) as data:
            if "x0" not in data.files:
                raise KeyError(f"Purified file has no x0 array: {path}")
            x0 = data["x0"].astype(np.float32)
            if x0.shape != (2, self.crop_size, self.crop_size):
                raise ValueError(f"Unexpected purified x0 shape in {path}: {x0.shape}")
            if not np.isfinite(x0).all():
                raise ValueError(f"Non-finite purified x0 in {path}")
            if "source" in data.files and str(data["source"]) != fname:
                raise ValueError(
                    f"Purified source mismatch in {path}: {str(data['source'])!r} != {fname!r}"
                )
            return torch.from_numpy(x0)

    def _maybe_load_measurement(
        self, fname: str, expected_shape: torch.Size
    ) -> Optional[torch.Tensor]:
        if self.measurement_dir is None:
            return None
        path = self.measurement_dir / f"{Path(fname).stem}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"measurement_dir was specified but the matched file is missing: {path}"
            )
        with np.load(path) as data:
            if "measured_kspace" not in data.files:
                raise KeyError(f"Measurement file has no measured_kspace array: {path}")
            measured = data["measured_kspace"].astype(np.float32)
            if tuple(measured.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Measurement shape mismatch in {path}: {measured.shape} != {tuple(expected_shape)}"
                )
            if not np.isfinite(measured).all():
                raise ValueError(f"Non-finite measured_kspace in {path}")
            if "source" in data.files and str(data["source"]) != fname:
                raise ValueError(f"Measurement source mismatch in {path}")
            return torch.from_numpy(measured)

    def __getitem__(self, index: int):
        path = self.files[index]
        kspace, smaps, _ = self._load_npz(path)
        mask, coil_mask = self._mask(index, kspace.shape[0])

        saved_measurement = self._maybe_load_measurement(path.name, kspace.shape)
        if saved_measurement is not None:
            measured_kspace = saved_measurement
        elif self.noise_std > 0:
            gen = torch.Generator().manual_seed(self.seed * 100000 + index)
            noise = torch.randn(kspace.shape, generator=gen, dtype=kspace.dtype) * self.noise_std
            measured_kspace = kspace + noise
        else:
            measured_kspace = kspace

        adjoint = networks.OPAT2(smaps)
        input_image = adjoint(measured_kspace, mask)
        target = adjoint(kspace, torch.ones_like(mask))
        purified = self._maybe_load_purified(path.name)
        if purified is not None:
            input_image = purified

        return input_image.float(), target.float(), smaps, mask, coil_mask, kspace, measured_kspace.float(), path.name


def split_files(
    data_root: os.PathLike | str = DEFAULT_KSPACE_DIR,
    train_size: int = 3000,
    val_size: int = 20,
    test_size: int = 64,
    seed: int = 0,
) -> Tuple[list[Path], list[Path], list[Path]]:
    kspace_dir = _resolve_kspace_dir(data_root)
    files = sorted(kspace_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {kspace_dir}")
    required = train_size + val_size + test_size
    if len(files) < required:
        raise ValueError(
            f"Requested {train_size}/{val_size}/{test_size} train/val/test files "
            f"({required} total), but only found {len(files)} in {kspace_dir}"
        )
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.permutation(len(files))]
    train = files[:train_size]
    val_start = len(train)
    val = files[val_start : val_start + val_size]
    test_start = val_start + len(val)
    test = files[test_start : test_start + test_size]
    return train, val, test


def require_matching_manifest(
    directory: os.PathLike | str,
    expected_files: Sequence[os.PathLike | str],
    artifact_label: str,
    allow_untracked: bool = False,
) -> Optional[dict]:
    """Require provenance and an exact ordered source-file match."""
    manifest_path = Path(directory) / "manifest.json"
    if not manifest_path.is_file():
        if allow_untracked:
            print(
                f"WARNING: using untracked {artifact_label} artifacts without "
                f"a manifest: {directory}"
            )
            return None
        raise FileNotFoundError(
            f"{artifact_label} directory has no manifest.json: {directory}. "
            "Regenerate it with the repaired pipeline, or explicitly use the "
            "allow-untracked option for exploratory (not paper-grade) use."
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = [Path(path).name for path in expected_files]
    actual = manifest.get("sources")
    if actual != expected:
        raise ValueError(
            f"{artifact_label} manifest sources do not exactly match the "
            f"requested split/order: expected {len(expected)}, "
            f"manifest has {len(actual) if isinstance(actual, list) else 'invalid'}"
        )
    return manifest


def require_manifest_settings(
    manifest: dict,
    expected: dict,
    artifact_label: str,
) -> None:
    """Fail if an artifact is relabeled as a different experiment."""
    actual = manifest.get("scientific_args", manifest.get("args", {}))
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{artifact_label} manifest settings mismatch: {mismatches}"
        )


def build_loaders(
    data_root: os.PathLike | str = DEFAULT_KSPACE_DIR,
    train_size: int = 3000,
    val_size: int = 20,
    test_size: int = 64,
    batch_size: int = 1,
    acceleration: float = 4.0,
    seed: int = 0,
    num_workers: int = 4,
    purified_dir: Optional[os.PathLike | str] = None,
    val_purified_dir: Optional[os.PathLike | str] = None,
    noise_std: float = 0.0,
    shift_fraction: float = 0.0,
):
    train_files, val_files, test_files = split_files(data_root, train_size, val_size, test_size, seed)
    train_ds = SMUGKspaceDataset(
        train_files,
        acceleration,
        seed=seed,
        purified_dir=purified_dir,
        noise_std=noise_std,
    )
    val_ds = SMUGKspaceDataset(
        val_files,
        acceleration,
        seed=seed + 10000,
        purified_dir=val_purified_dir,
    )
    test_ds = SMUGKspaceDataset(test_files, acceleration, seed=seed + 20000, shift_fraction=shift_fraction)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader


def loadData(
    Kspace_data_name=DEFAULT_KSPACE_DIR,
    mask_data_name=None,
    num_train: int = 3000,
    num_test: int = 20,
    batch_size: int = 1,
    targeted_acceleration: float = 4.0,
):
    """Backward-compatible wrapper used by older scripts."""
    del mask_data_name
    train_loader, val_loader, _ = build_loaders(
        data_root=Kspace_data_name,
        train_size=num_train,
        val_size=num_test,
        test_size=0,
        batch_size=batch_size,
        acceleration=targeted_acceleration,
    )
    return train_loader, val_loader
