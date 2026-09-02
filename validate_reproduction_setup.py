from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import global_network_dataset
from models.didn import DIDN
from train_MoDL import load_model_weights


KNOWN_SHA256 = {
    "DIDN_lambda1_3000_images_trained.pt":
        "4c1053994a0b7b6d93f342ec94ca5940f956c590f77ca3843f3ef8bc22733dc3",
    "checkpoint_95.pth":
        "1b65ca98820fdb9327d76e15a03eea336c3afb9edaa8966af3893c4134d25755",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Fail-fast validation for pretrained RODIO reproduction.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--modl-checkpoint", required=True)
    p.add_argument("--score-checkpoint", required=True)
    p.add_argument("--train-size", type=int, default=3000)
    p.add_argument("--val-size", type=int, default=20)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scan-files", type=int, default=3)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def inspect_raw(path: Path):
    with np.load(path) as data:
        required = {"s_r", "s_i", "k_r", "k_i"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        arrays = {key: data[key] for key in required}
    shapes = {key: list(value.shape) for key, value in arrays.items()}
    dtypes = {key: str(value.dtype) for key, value in arrays.items()}
    max_abs = {key: float(np.max(np.abs(value))) for key, value in arrays.items()}
    for key, value in arrays.items():
        if value.ndim != 3 or value.shape[0] != 15:
            raise ValueError(f"{path}:{key} expected [15,H,W], got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{path}:{key} contains non-finite values")
        if max_abs[key] > 32767.01:
            raise ValueError(
                f"{path}:{key} exceeds int16 scaling range: max_abs={max_abs[key]}"
            )
    return {"file": path.name, "shapes": shapes, "dtypes": dtypes, "max_abs": max_abs}


def main():
    args = parse_args()
    modl_path, score_path = Path(args.modl_checkpoint), Path(args.score_checkpoint)
    for path in (modl_path, score_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train, val, test = global_network_dataset.split_files(
        args.data_root, args.train_size, args.val_size, args.test_size, args.seed
    )
    all_files = train + val + test
    raw = [inspect_raw(path) for path in all_files[:max(1, args.scan_files)]]

    # Exercise the complete preprocessing path once per split.
    loaded = {}
    for name, files, split_seed in (
        ("train", train, args.seed),
        ("val", val, args.seed + 10000),
        ("test", test, args.seed + 20000),
    ):
        if not files:
            loaded[name] = None
            continue
        sample = global_network_dataset.SMUGKspaceDataset(
            files[:1], acceleration=4.0, seed=split_seed
        )[0]
        loaded[name] = {
            "source": sample[-1],
            "input_shape": list(sample[0].shape),
            "target_shape": list(sample[1].shape),
            "smap_shape": list(sample[2].shape),
            "kspace_shape": list(sample[5].shape),
            "finite": all(bool(torch.isfinite(x).all()) for x in sample[:7]),
        }

    model = DIDN(
        2, 2, num_chans=64, pad_data=True, global_residual=True, n_res_blocks=2
    ).float()
    load_model_weights(model, modl_path)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    # Check score checkpoint structure without allocating the second model copy.
    try:
        score_state = torch.load(score_path, map_location="cpu", weights_only=False)
    except TypeError:
        score_state = torch.load(score_path, map_location="cpu")
    required_score_keys = {"model", "ema", "optimizer", "step"}
    missing_score = required_score_keys.difference(score_state)
    if missing_score:
        raise KeyError(f"Score checkpoint missing {sorted(missing_score)}")

    hashes = {path.name: sha256_file(path) for path in (modl_path, score_path)}
    hash_match = {
        name: (KNOWN_SHA256.get(name) is None or value == KNOWN_SHA256[name])
        for name, value in hashes.items()
    }
    result = {
        "status": "PASS",
        "split_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "first_sources": {
            "train": train[0].name if train else None,
            "val": val[0].name if val else None,
            "test": test[0].name if test else None,
        },
        "raw_examples": raw,
        "loaded_examples": loaded,
        "modl_parameter_count": parameter_count,
        "score_step": int(score_state["step"]),
        "score_model_tensor_count": len(score_state["model"]),
        "sha256": hashes,
        "known_hash_match": hash_match,
        "normalization_contract": (
            "s_r/s_i/k_r/k_i are interpreted as int16-scale values and divided by 32767"
        ),
    }
    rendered = json.dumps(result, indent=2)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
