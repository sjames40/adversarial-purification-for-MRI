# Robust Physics-Based Deep MRI Reconstruction via Diffusion Purification (RODIO)

[![IEEE TNNLS](https://img.shields.io/badge/IEEE-TNNLS-blue.svg)](https://ieeexplore.ieee.org/document/11352979)
[![arXiv](https://img.shields.io/badge/arXiv-2309.05794-b31b1b.svg)](https://arxiv.org/abs/2309.05794)

Research code and reproduction workflow for the paper:

### Robust Physics-Based Deep MRI Reconstruction via Diffusion Purification

**IEEE Transactions on Neural Networks and Learning Systems**, vol. 37, no. 5, pp. 2347–2361, 2026.

Paper links: [IEEE Xplore](https://ieeexplore.ieee.org/document/11352979) | [arXiv:2309.05794](https://arxiv.org/abs/2309.05794)

---

## Overview

This repository implements **RODIO**, a robustification framework for physics-based deep MRI reconstruction. RODIO uses a pretrained score-based diffusion model to purify a potentially corrupted MRI reconstruction before passing it through a fine-tuned deep unrolled reconstruction network.

RODIO is designed to improve reconstruction robustness to measurement perturbations, changes in acceleration factor, sampling-mask shifts, and other distribution shifts.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sjames40/adversarial-purification-for-MRI.git
cd adversarial-purification-for-MRI
```

### 2. Create an environment

```bash
conda create -n rodio python=3.10 -y
conda activate rodio
```

Install a PyTorch build compatible with your CUDA driver and GPU, then install the remaining packages:

```bash
pip install torch torchvision
pip install numpy scipy scikit-image h5py pillow matplotlib tqdm sigpy packaging ml-collections ninja
```

`ninja` is recommended for compiling fused CUDA operations. Without it, the code can fall back to native PyTorch operations but purification may be slower.

### 3. Download pretrained checkpoints

Create the weights directory:

```bash
mkdir -p weights
```

Download the pretrained score-based diffusion model:

```bash
wget -O weights/checkpoint_95.pth \
  "https://www.dropbox.com/s/27gtxkmh2dlkho9/checkpoint_95.pth?dl=1"
```

Download the pretrained vanilla MoDL/DIDN model:

```bash
wget -O weights/DIDN_lambda1_3000_images_trained.pt \
  "https://www.dropbox.com/scl/fi/xnlrcexczb8yr3neshj1b/DIDN_lambda1_3000_images_trained_2.pt?rlkey=icrqens0ltzvtjusxyzuck89s&dl=1"
```

The diffusion checkpoint was adopted from the score-based MRI reconstruction work described in [arXiv:2110.05243](https://arxiv.org/abs/2110.05243).

Expected checkpoint layout:

```text
weights/
├── checkpoint_95.pth
└── DIDN_lambda1_3000_images_trained.pt
```

---

## Dataset

RODIO uses multicoil knee MRI data from the [fastMRI dataset](https://fastmri.med.nyu.edu/).

The current loader expects preprocessed `.npz` files rather than raw fastMRI HDF5 files. Organize the data as:

```text
data/
└── NEW_KSPACE/
    ├── sample_0001.npz
    ├── sample_0002.npz
    └── ...
```

---

# Reproduction Pipeline

Set the project paths before running the commands:

```bash
export RODIO_ROOT=/path/to/adversarial-purification-for-MRI
export RODIO_DATA=/path/to/data
export RODIO_SCORE="$RODIO_ROOT/weights/checkpoint_95.pth"
export RODIO_MODL="$RODIO_ROOT/weights/DIDN_lambda1_3000_images_trained.pt"
export RODIO_RUNS="$RODIO_ROOT/runs"

cd "$RODIO_ROOT"
mkdir -p "$RODIO_RUNS"
```

## Step 1: Validate the setup

```bash
python validate_reproduction_setup.py \
  --data-root "$RODIO_DATA" \
  --modl-checkpoint "$RODIO_MODL" \
  --score-checkpoint "$RODIO_SCORE" \
  --train-size 3000 \
  --val-size 20 \
  --test-size 64 \
  --seed 0 \
  --scan-files 10 \
  --output-json "$RODIO_RUNS/setup_validation.json"
```

## Step 2: Evaluate vanilla MoDL

```bash
python evaluate_modl.py \
  --data-root "$RODIO_DATA" \
  --checkpoint "$RODIO_MODL" \
  --checkpoint-kind vanilla \
  --output-json "$RODIO_RUNS/eval_vanilla_clean_4x.json" \
  --train-size 3000 \
  --val-size 20 \
  --test-size 64 \
  --acceleration 4 \
  --block-iter 6 \
  --lambda-reg 1 \
  --cg-tol 1e-6 \
  --seed 0 \
  --device cuda:0
```

## Step 3: Generate purified training and validation examples

Generate the 3000 purified training examples:

```bash
python rodio_purify.py \
  --data-root "$RODIO_DATA" \
  --score-checkpoint "$RODIO_SCORE" \
  --output-dir "$RODIO_RUNS/purified_train_sigmaft001_pst150" \
  --split train \
  --train-size 3000 \
  --val-size 20 \
  --test-size 64 \
  --acceleration 4 \
  --pst-step 150 \
  --num-scales 500 \
  --sigma-ft 0.01 \
  --snr 0.16 \
  --corrector-steps 1 \
  --seed 0 \
  --device cuda:0
```

Generate the 20 purified validation examples:

```bash
python rodio_purify.py \
  --data-root "$RODIO_DATA" \
  --score-checkpoint "$RODIO_SCORE" \
  --output-dir "$RODIO_RUNS/purified_val_sigma001_pst150" \
  --split val \
  --train-size 3000 \
  --val-size 20 \
  --test-size 64 \
  --acceleration 4 \
  --pst-step 150 \
  --num-scales 500 \
  --noise-std 0.01 \
  --snr 0.16 \
  --corrector-steps 1 \
  --seed 0 \
  --device cuda:0
```

## Step 4: Fine-tune MoDL

Initialize from the pretrained vanilla MoDL checkpoint and fine-tune on purified examples:

```bash
python train_MoDL.py \
  --data-root "$RODIO_DATA" \
  --checkpoints-dir "$RODIO_ROOT/weights" \
  --name rodio_ft_sigma001_pst150 \
  --train-size 3000 \
  --val-size 20 \
  --test-size 64 \
  --batch-size 1 \
  --num-workers 4 \
  --epochs 20 \
  --lr 1e-4 \
  --lr-decay-start 10 \
  --acceleration 4 \
  --block-iter 6 \
  --lambda-reg 1 \
  --cg-tol 1e-6 \
  --init-weights "$RODIO_MODL" \
  --purified-dir "$RODIO_RUNS/purified_train_sigmaft001_pst150" \
  --val-purified-dir "$RODIO_RUNS/purified_val_sigma001_pst150" \
  --expected-sigma-ft 0.01 \
  --expected-val-noise-std 0.01 \
  --seed 0 \
  --gpu-ids 0
```

The output checkpoint is:

```text
weights/rodio_ft_sigma001_pst150/vali_best.pth
```

## Step 5: Generate clean test purification

```bash
python rodio_purify.py \
  --data-root "$RODIO_DATA" \
  --score-checkpoint "$RODIO_SCORE" \
  --output-dir "$RODIO_RUNS/purified_test_clean_4x_pst150" \
  --split test \
  --train-size 3000 \
  --val-size 20 \
  --test-size 64 \
  --acceleration 4 \
  --pst-step 150 \
  --num-scales 500 \
  --noise-std 0 \
  --snr 0.16 \
  --corrector-steps 1 \
  --seed 0 \
  --device cuda:0
```

## Step 6: Evaluate RODIO

```bash
python evaluate_modl.py \
  --data-root "$RODIO_DATA" \
  --checkpoint "$RODIO_ROOT/weights/rodio_ft_sigma001_pst150/vali_best.pth" \
  --checkpoint-kind rodio_finetuned \
  --purified-dir "$RODIO_RUNS/purified_test_clean_4x_pst150" \
  --output-json "$RODIO_RUNS/eval_rodio_clean_4x.json" \
  --train-size 3000 \
  --val-size 20 \
  --test-size 64 \
  --acceleration 4 \
  --noise-std 0 \
  --block-iter 6 \
  --lambda-reg 1 \
  --cg-tol 1e-6 \
  --seed 0 \
  --device cuda:0
```

### Evaluate standalone diffusion purification

```bash
python evaluate_purified.py \
  --data-root "$RODIO_DATA" \
  --purified-dir "$RODIO_RUNS/purified_test_clean_4x_pst150" \
  --output-json "$RODIO_RUNS/eval_dp_clean_4x.json" \
  --train-size 3000 \
  --val-size 20 \
  --test-size 64 \
  --acceleration 4 \
  --noise-std 0 \
  --seed 0 \
  --device cuda:0
```

---

## PGD Robustness Evaluation

Generate a 30-step measurement-space PGD attack with `epsilon = 0.004`:

```bash
python generate_kspace_attack.py \
  --data-root "$RODIO_DATA" \
  --checkpoint "$RODIO_MODL" \
  --output-dir "$RODIO_RUNS/attack_test_pgd_eps0004" \
  --split test \
  --train-size 3000 \
  --val-size 20 \
  --test-size 64 \
  --acceleration 4 \
  --method pgd \
  --reference clean_reconstruction \
  --loss-domain complex \
  --epsilon 0.004 \
  --steps 30 \
  --step-size 0.0013333333 \
  --block-iter 6 \
  --lambda-reg 1 \
  --cg-tol 1e-6 \
  --seed 0 \
  --device cuda:0
```

---

## Faster Diffusion Reconstruction

For faster diffusion-based MRI reconstruction and purification, see the [DDS repository](https://github.com/hyungjin-chung/DDS). DDS uses a different accelerated diffusion solver and should be reported separately from the predictor-corrector RODIO implementation.

---

## Citation

If you use this code, please cite the RODIO paper:

```bibtex
@ARTICLE{RODIO,
  author={Alkhouri, Ismail R. and Liang, Shijun and Wang, Rongrong and Qu, Qing and Ravishankar, Saiprasad},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Robust Physics-Based Deep MRI Reconstruction via Diffusion Purification},
  year={2026},
  volume={37},
  number={5},
  pages={2347-2361},
  doi={10.1109/TNNLS.2025.3631742}
}
```

---

## Acknowledgements

This project uses the [fastMRI dataset](https://fastmri.med.nyu.edu/) and a pretrained score-based MRI model derived from [score-MRI](https://arxiv.org/abs/2110.05243). Please cite the corresponding works and follow their data and software licenses.
