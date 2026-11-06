# Method names

| Flag | What |
|------|------|
| `3dgs` | Official 3D Gaussian Splatting (Kerbl et al., TOG 2023). External trainer. |
| `baseline` | Scaffold-GS |
| `moe` | Flat MoE (8 experts) |
| `moe_gnn` | MoE + GNN (warm-start from `moe`) |
| `moe_tree` | Growing Tree MoE, no GNN (start 8 = MoE 1:1, grow hard leaves, color-first) |
| `tree_gnn` | Growing Tree MoE + GNN (same grow recipe as `moe_tree`) |

```bash
# single scene
bash train_method.sh --method 3dgs     -d mipnerf360/kitchen -g 0
bash train_method.sh --method moe      -d mipnerf360/kitchen -g 0
bash train_method.sh --method moe_gnn  -d mipnerf360/kitchen -g 0   # needs moe
bash train_method.sh --method moe_tree -d mipnerf360/kitchen -g 0   # needs moe
bash train_method.sh --method tree_gnn -d mipnerf360/kitchen -g 0   # needs moe

# 3D-GS on every suite in this repo (skips finished runs)
bash sweep.sh --suite all --methods 3dgs --skip-done -g 0
# paper 19-scene split (no BungeeNeRF)
bash sweep.sh --suite paper --methods 3dgs --skip-done -g 0

# full suite (skips scenes/methods that already have results)
bash sweep.sh --suite mip360 --skip-done -g 0
```

Scaffold-GS helpers (upstream, keep): `train.py`, `render.py`, `metrics.py`, `convert.py`
