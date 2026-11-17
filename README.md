# Growing Tree MoE for Scaffold-GS

Code release for our CVPR submission: a **Growing Tree Mixture-of-Experts** decoder for [Scaffold-GS](https://github.com/city-super/Scaffold-GS) attribute prediction (opacity / covariance / color).

This repository contains the **training and rendering code** only. Paper sources, ablation plots, and internal analysis scripts are not included.

## Setup

```bash
conda create -n scaffold_gs python=3.11 -y
conda activate scaffold_gs
pip install torch torchvision  # match your CUDA
pip install -r requirements.txt
```

Build the differential Gaussian rasterizer submodule (same as Scaffold-GS):

```bash
pip install -e submodules/diff_gaussian_rasterization_df
pip install -e submodules/simple-knn
```

Place datasets under `data/` (Mip-NeRF 360, Tanks & Temples, Deep Blending, NeRF-Synthetic, etc.).

## Training

Use `train_method.sh` to run the main recipes:

```bash
# Scaffold-GS baseline (shared MLP)
bash train_method.sh --method baseline -d mipnerf360/kitchen -g 0

# Flat MoE (Stage A)
bash train_method.sh --method moe -d mipnerf360/kitchen -g 0

# Growing Tree MoE (Stage B; warm-starts from a finished flat MoE run)
bash train_method.sh --method moe_tree -d mipnerf360/kitchen -g 0
```

Or call `train.py` directly. Checkpoints and logs are written under `outputs/`.

## Evaluation

```bash
python render.py -m outputs/<dataset>/<method>/<timestamp>
python metrics.py -m outputs/<dataset>/<method>/<timestamp>
```

## Citation

If you use this code, please cite our paper (bibtex will be added upon acceptance / arXiv).

## Acknowledgments

Built on [Scaffold-GS](https://github.com/city-super/Scaffold-GS) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
