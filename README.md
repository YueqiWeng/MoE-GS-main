# Learning to Grow Experts: Dynamic Tree MoE for Scaffold-GS

Official code for **Growing Tree Mixture-of-Experts (Tree MoE)** — a hierarchical expert decoder for [Scaffold-GS](https://github.com/city-super/Scaffold-GS) attribute prediction (opacity / covariance / color).

> This repository contains **training and rendering code only**.  
> Paper sources, figures, and internal analysis scripts are not included.

## Method (short)

1. **Stage A — Flat MoE.** Train a sparsely gated MoE decoder (default: 8 experts, Top-2) with a dense-to-sparse curriculum.
2. **Stage B — Growing Tree MoE.** Warm-start a tree from the flat experts, then selectively split hard leaves using routing traffic, reconstruction loss, and feature variation.

Entry point: `train_method.sh` (or `train.py`).

## Requirements

- Linux + NVIDIA GPU
- CUDA toolkit matching your PyTorch build
- Conda (recommended)

## Installation

```bash
git clone https://github.com/YueqiWeng/MoE-GS-main.git
cd MoE-GS-main

conda create -n tree_moe python=3.11 -y
conda activate tree_moe

# Install PyTorch for your CUDA version: https://pytorch.org
pip install torch torchvision

pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__version__)").html
```

Install the Scaffold-GS CUDA extensions (not vendored here):

```bash
# Rasterizer (provides `diff_gaussian_rasterization`)
git clone https://github.com/city-super/Scaffold-GS.git --recursive /tmp/Scaffold-GS
pip install -e /tmp/Scaffold-GS/submodules/diff-gaussian-rasterization
pip install -e /tmp/Scaffold-GS/submodules/simple-knn
```

Alternatively, clone those two submodules into a local `submodules/` folder and `pip install -e` them the same way.

## Data

Put scenes under `data/`, e.g.:

```text
data/mipnerf360/kitchen/
data/db/playroom/
data/tandt/truck/
```

Use the same COLMAP / Scaffold-GS layout as the original Scaffold-GS release.

## Training

```bash
# Scaffold-GS baseline (shared MLP)
bash train_method.sh --method baseline -d mipnerf360/kitchen -g 0

# Flat MoE (Stage A)
bash train_method.sh --method moe -d mipnerf360/kitchen -g 0

# Growing Tree MoE (Stage B; warm-starts from a finished `moe` run)
bash train_method.sh --method moe_tree -d mipnerf360/kitchen -g 0
```

Outputs are written to `outputs/<dataset>/<method>/<timestamp>/`.

Useful flags (see `train_method.sh`):

- `--num_experts`, `--moe_top_k`
- `--moe_max_total_grows` / growth schedule knobs for Tree MoE
- `--no_warm_start` to train the tree without flat-MoE initialization

## Rendering & metrics

```bash
python render.py -m outputs/mipnerf360/kitchen/moe_tree/<timestamp>
python metrics.py -m outputs/mipnerf360/kitchen/moe_tree/<timestamp>
```

## Repository layout

```text
train.py / train_method.sh   # training
render.py / metrics.py       # evaluation
scene/                       # Scaffold-GS model + MoE / Tree MoE / optional GNN
gaussian_renderer/           # rasterize + prefilter
arguments/ / utils/          # configs and helpers
```

## Citation

```bibtex
@inproceedings{treemoe2027,
  title     = {Learning to Grow Experts: Dynamic Tree Mixture-of-Experts
               for Anchor-Based 3D Gaussian Scene Reconstruction},
  author    = {Anonymous},
  booktitle = {CVPR},
  year      = {2027}
}
```

## Acknowledgments

Built on [Scaffold-GS](https://github.com/city-super/Scaffold-GS) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). Please respect their licenses (including the Inria / MPII research license for Gaussian Splatting components).
