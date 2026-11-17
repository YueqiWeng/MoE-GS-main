# Learning to Grow Experts: Dynamic Tree MoE for Scaffold-GS

Code for Growing Tree MoE on [Scaffold-GS](https://github.com/city-super/Scaffold-GS).

## Install

```bash
conda create -n tree_moe python=3.11 -y && conda activate tree_moe
pip install torch torchvision   # match your CUDA: https://pytorch.org
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__version__)").html

# CUDA extensions (from Scaffold-GS)
git clone https://github.com/city-super/Scaffold-GS.git --recursive /tmp/Scaffold-GS
pip install -e /tmp/Scaffold-GS/submodules/diff-gaussian-rasterization
pip install -e /tmp/Scaffold-GS/submodules/simple-knn
```

## Train / eval

Put data under `data/` (Scaffold-GS layout).

```bash
bash train_method.sh --method baseline -d mipnerf360/kitchen -g 0   # shared MLP
bash train_method.sh --method moe      -d mipnerf360/kitchen -g 0   # flat MoE
bash train_method.sh --method moe_tree -d mipnerf360/kitchen -g 0   # Tree MoE

python render.py  -m outputs/mipnerf360/kitchen/moe_tree/<timestamp>
python metrics.py -m outputs/mipnerf360/kitchen/moe_tree/<timestamp>
```
