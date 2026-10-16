# Flat MoE ablation: N × Top-k (all scenes)

PSNR at 30k. Default recipe is **N=8, Top-2** (`moe`).

| Scene | N=4 Top-1 | N=4 Top-2 | N=8 Top-1 | N=8 Top-2 | Best |
|---|---:|---:|---:|---:|---|
| bungeenerf/amsterdam | 27.059 | 27.071 | 26.994 | 27.037 | 4/2 |
| bungeenerf/barcelona | 26.831 | 26.816 | 26.890 | 26.907 | 8/2 |
| bungeenerf/bilbao | 28.208 | 28.238 | 28.212 | 28.166 | 4/2 |
| bungeenerf/chicago | 22.230 | 21.982 | 21.188 | 21.269 | 4/1 |
| bungeenerf/hollywood | 23.935 | 23.751 | 23.792 | 23.741 | 4/1 |
| bungeenerf/pompidou | 19.994 | 19.936 | 19.650 | 19.634 | 4/1 |
| bungeenerf/quebec | 28.708 | 28.692 | 28.507 | 28.553 | 4/1 |
| bungeenerf/rome | 23.238 | 23.238 | 23.367 | 23.106 | 8/1 |
| db/drjohnson | 29.702 | 29.676 | 29.791 | 29.736 | 8/1 |
| db/playroom | 31.017 | 31.003 | 30.931 | 31.021 | 8/2 |
| mipnerf360/bicycle | 25.195 | 25.185 | 25.232 | 25.268 | 8/2 |
| mipnerf360/bonsai | 32.716 | 32.750 | 33.095 | 33.071 | 8/1 |
| mipnerf360/counter | 29.466 | 29.487 | 29.564 | 29.658 | 8/2 |
| mipnerf360/garden | 27.387 | 27.399 | 27.424 | 27.436 | 8/2 |
| mipnerf360/kitchen | 31.628 | 31.842 | 31.776 | 31.713 | 4/2 |
| mipnerf360/room | 32.253 | 32.177 | 32.119 | 32.142 | 4/1 |
| mipnerf360/stump | 26.667 | 26.672 | 26.607 | 26.661 | 4/2 |
| nerfsynthetic/chair | 34.723 | 34.675 | 34.931 | 34.900 | 8/1 |
| nerfsynthetic/drums | 25.925 | 25.949 | 26.167 | 26.156 | 8/1 |
| nerfsynthetic/ficus | 34.358 | 34.230 | 34.268 | 34.196 | 4/1 |
| nerfsynthetic/hotdog | 37.667 | 37.574 | 37.607 | 37.744 | 8/2 |
| nerfsynthetic/lego | 35.286 | 35.227 | 35.474 | 35.519 | 8/2 |
| nerfsynthetic/materials | 30.054 | 30.140 | 30.048 | 30.167 | 8/2 |
| nerfsynthetic/mic | 36.269 | 36.132 | 36.247 | 36.125 | 4/1 |
| nerfsynthetic/ship | 31.195 | 31.183 | 31.299 | 31.263 | 8/1 |
| tandt/train | 22.431 | 22.621 | 22.256 | 22.695 | 8/2 |
| tandt/truck | 25.987 | 25.963 | 25.960 | 26.035 | 8/2 |
| **Mean (27)** | **28.894** | **28.874** | **28.867** | **28.886** | |

## Wins (highest PSNR per scene)
- N=8 Top-2: **10/27**
- N=4 Top-1: **7/27**
- N=8 Top-1: **6/27**
- N=4 Top-2: **4/27**

