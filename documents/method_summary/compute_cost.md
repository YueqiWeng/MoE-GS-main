# Computation cost: moe_tree vs baselines

Metrics: **training wall-clock** (all finished runs) and **peak GPU / CPU memory during rendering** (5 representative scenes).

## Mean summary

| Method | Train time (27 scenes) | Peak GPU (5 scenes) | Peak CPU RSS (5 scenes) |
|---|---|---|---|
| baseline | 16.1 min | 3.64 GB | 1.59 GB |
| moe | 30.8 min | 4.04 GB | 1.63 GB |
| **moe_tree** | **41.2 min** | **3.78 GB** | **1.66 GB** |

Relative to baseline: moe_tree trains **2.6×** longer; GPU/CPU memory nearly the same as flat MoE.

## Per-scene (measured memory)

| Scene | Method | Train (min) | Peak GPU (GB) | Peak CPU (GB) |
|---|---|---|---|---|
| mipnerf360/kitchen | baseline | 25.8 | 6.124 | 1.245 |
| mipnerf360/kitchen | moe | 49.5 | 6.765 | 1.492 |
| mipnerf360/kitchen | moe_tree | 58.4 | 6.381 | 1.628 |
| mipnerf360/bicycle | baseline | 22.8 | 4.879 | 1.628 |
| mipnerf360/bicycle | moe | 38.9 | 5.542 | 1.583 |
| mipnerf360/bicycle | moe_tree | 57.8 | 5.044 | 1.582 |
| tandt/truck | baseline | 10.0 | 1.962 | 1.63 |
| tandt/truck | moe | 21.6 | 2.361 | 1.635 |
| tandt/truck | moe_tree | 28.2 | 2.033 | 1.635 |
| nerfsynthetic/lego | baseline | 4.7 | 2.353 | 1.85 |
| nerfsynthetic/lego | moe | 13.1 | 2.525 | 1.819 |
| nerfsynthetic/lego | moe_tree | 18.8 | 2.435 | 1.847 |
| db/playroom | baseline | 12.5 | 2.889 | 1.599 |
| db/playroom | moe | 16.6 | 2.99 | 1.599 |
| db/playroom | moe_tree | 21.9 | 3.008 | 1.595 |
