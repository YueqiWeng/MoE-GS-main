# Sensitivity: $\lambda_{\mathrm{load}}$

CSV: [ablate_lambda_load_all_scenes.csv](ablate_lambda_load_all_scenes.csv)

Warm-started Tree MoE; only `lambda_load` changes. `0` = `ablate_noload`, `0.005` = Full `moe_tree`.

## Completion

| $\lambda_{\mathrm{load}}$ | Done |
|---:|---:|
| 0 | 19/19 |
| 0.001 | 1/19 |
| 0.005 | 19/19 |
| 0.01 | 0/19 |
| 0.05 | 0/19 |

## Mean PSNR

| Split | $\lambda=0$ | $\lambda=0.001$ | $\lambda=0.005$ | $\lambda=0.01$ | $\lambda=0.05$ |
|---|---:|---:|---:|---:|---:|
| mipnerf360 | 29.805 | 25.463 | 29.796 | - | - |
| tandt | 24.532 | - | 24.523 | - | - |
| db | 30.505 | - | 30.500 | - | - |
| nerfsynthetic | 33.555 | - | 33.573 | - | - |
| **all** | 30.903 | 25.463 | 30.906 | - | - |
