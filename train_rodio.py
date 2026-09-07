"""Prepare purified examples and fine-tune MoDL with one command.

This is the recommended entry point for RODIO training.  It deliberately runs
purification and fine-tuning in separate child processes: the score model can
be released before MoDL training starts, while the complete workflow remains a
single user-facing Python command.

The paper/reproduction defaults live below.  Most users only need to provide
``--data-root`` when their data are not at ``DEFAULT_DATA_ROOT``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import torch

import global_network_dataset


# ---------------------------------------------------------------------------
# RODIO paper/reproduction defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = global_network_dataset.DEFAULT_DATA_ROOT
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_SCORE_CHECKPOINT = DEFAULT_WEIGHTS_DIR / "checkpoint_95.pth"
DEFAULT_MODL_CHECKPOINT = (
    DEFAULT_WEIGHTS_DIR / "DIDN_lambda1_3000_images_trained.pt"
)

TRAIN_SIZE = 3000
VAL_SIZE = 20
TEST_SIZE = 64
ACCELERATION = 4.0
PST_STEP = 150
NUM_SCALES = 500
SIGMA_FT = 0.01
SNR = 0.16
CORRECTOR_STEPS = 1
SEED = 0

BATCH_SIZE = 1
PURIFICATION_WORKERS = 2
TRAIN_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
LR_DECAY_START = 10
MODL_UNROLLS = 6
LAMBDA_REG = 1.0
CG_TOL = 1e-6

DEFAULT_EXPERIMENT_NAME = "rodio_ft_sigma001_pst150"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RODIO train/validation purifications and fine-tune MoDL. "
            "Scientific defaults are defined at the top of this file."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--score-checkpoint", default=None)
    parser.add_argument("--modl-checkpoint", default=None)
    parser.add_argument("--name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Purification device; the same GPU is used for fine-tuning.",
    )
    parser.add_argument(
        "--skip-purification",
        action="store_true",
        help="Use already completed, manifest-tracked train/val purifications.",
    )
    parser.add_argument(
        "--overwrite-purification",
        action="store_true",
        help="Regenerate matching purification files. Existing weights are never overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the three commands without running the long experiment.",
    )
    return parser.parse_args()


def _absolute(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _training_gpu_ids(device: str) -> str:
    if device == "cpu":
        return "-1"
    if device.startswith("cuda:") and device[5:].isdigit():
        return device[5:]
    raise ValueError("--device must be 'cpu' or an indexed CUDA device such as 'cuda:0'")


def build_commands(args: argparse.Namespace) -> tuple[list[list[str]], dict]:
    if Path(args.name).name != args.name:
        raise ValueError("--name must be a single experiment-directory name")
    data_root = _absolute(args.data_root)
    weights_dir = _absolute(args.weights_dir)
    runs_dir = _absolute(args.runs_dir)
    score_checkpoint = _absolute(
        args.score_checkpoint or weights_dir / "checkpoint_95.pth"
    )
    modl_checkpoint = _absolute(
        args.modl_checkpoint
        or weights_dir / "DIDN_lambda1_3000_images_trained.pt"
    )
    train_purified = runs_dir / "purified_train_sigmaft001_pst150"
    val_purified = runs_dir / "purified_val_sigma001_pst150"
    python = sys.executable

    shared_purification = [
        "--data-root", str(data_root),
        "--score-checkpoint", str(score_checkpoint),
        "--train-size", str(TRAIN_SIZE),
        "--val-size", str(VAL_SIZE),
        "--test-size", str(TEST_SIZE),
        "--acceleration", str(ACCELERATION),
        "--pst-step", str(PST_STEP),
        "--num-scales", str(NUM_SCALES),
        "--snr", str(SNR),
        "--corrector-steps", str(CORRECTOR_STEPS),
        "--batch-size", str(BATCH_SIZE),
        "--num-workers", str(PURIFICATION_WORKERS),
        "--seed", str(SEED),
        "--device", args.device,
    ]
    existing_policy = (
        ["--overwrite"] if args.overwrite_purification else ["--skip-existing"]
    )
    train_purify_command = [
        python, str(PROJECT_ROOT / "rodio_purify.py"),
        "--output-dir", str(train_purified),
        "--split", "train",
        "--sigma-ft", str(SIGMA_FT),
        *shared_purification,
        *existing_policy,
    ]
    val_purify_command = [
        python, str(PROJECT_ROOT / "rodio_purify.py"),
        "--output-dir", str(val_purified),
        "--split", "val",
        "--noise-std", str(SIGMA_FT),
        *shared_purification,
        *existing_policy,
    ]
    train_command = [
        python, str(PROJECT_ROOT / "train_MoDL.py"),
        "--data-root", str(data_root),
        "--checkpoints-dir", str(weights_dir),
        "--name", args.name,
        "--train-size", str(TRAIN_SIZE),
        "--val-size", str(VAL_SIZE),
        "--test-size", str(TEST_SIZE),
        "--batch-size", str(BATCH_SIZE),
        "--num-workers", str(TRAIN_WORKERS),
        "--epochs", str(EPOCHS),
        "--lr", str(LEARNING_RATE),
        "--lr-decay-start", str(LR_DECAY_START),
        "--acceleration", str(ACCELERATION),
        "--block-iter", str(MODL_UNROLLS),
        "--lambda-reg", str(LAMBDA_REG),
        "--cg-tol", str(CG_TOL),
        "--init-weights", str(modl_checkpoint),
        "--purified-dir", str(train_purified),
        "--val-purified-dir", str(val_purified),
        "--expected-sigma-ft", str(SIGMA_FT),
        "--expected-val-noise-std", str(SIGMA_FT),
        "--seed", str(SEED),
        "--gpu-ids", _training_gpu_ids(args.device),
    ]
    commands = [train_purify_command, val_purify_command, train_command]
    resolved = {
        "data_root": str(data_root),
        "weights_dir": str(weights_dir),
        "runs_dir": str(runs_dir),
        "score_checkpoint": str(score_checkpoint),
        "modl_checkpoint": str(modl_checkpoint),
        "train_purified_dir": str(train_purified),
        "val_purified_dir": str(val_purified),
        "experiment_dir": str(weights_dir / args.name),
        "output_checkpoint": str(weights_dir / args.name / "vali_best.pth"),
    }
    return commands, resolved


def validate_inputs(resolved: dict) -> None:
    # split_files checks both the directory layout and the required sample count.
    global_network_dataset.split_files(
        resolved["data_root"], TRAIN_SIZE, VAL_SIZE, TEST_SIZE, SEED
    )
    for key in ("score_checkpoint", "modl_checkpoint"):
        path = Path(resolved[key])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key}: {path}")
    experiment_dir = Path(resolved["experiment_dir"])
    tracked_outputs = ("args.json", "history.json", "latest.pth", "vali_best.pth")
    if any((experiment_dir / filename).exists() for filename in tracked_outputs):
        raise FileExistsError(
            f"Training outputs already exist in {experiment_dir}; choose a new --name."
        )


def run_command(command: list[str], label: str) -> None:
    print(f"\n[{label}]\n{shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.skip_purification and args.overwrite_purification:
        raise ValueError(
            "--skip-purification and --overwrite-purification are mutually exclusive"
        )
    commands, resolved = build_commands(args)
    labels = (
        "1/3 purify training split",
        "2/3 purify validation split",
        "3/3 fine-tune MoDL",
    )

    print("Resolved RODIO training configuration:")
    print(json.dumps(resolved, indent=2))
    selected = [(commands[2], labels[2])] if args.skip_purification else list(zip(commands, labels))
    if args.dry_run:
        for command, label in selected:
            print(f"\n[{label}]\n{shlex.join(command)}")
        return

    validate_inputs(resolved)
    Path(resolved["runs_dir"]).mkdir(parents=True, exist_ok=True)
    for command, label in selected:
        run_command(command, label)

    output_checkpoint = Path(resolved["output_checkpoint"])
    if not output_checkpoint.is_file():
        raise RuntimeError(f"Training finished without expected checkpoint: {output_checkpoint}")
    print(f"\nRODIO fine-tuning complete: {output_checkpoint}")


if __name__ == "__main__":
    main()
