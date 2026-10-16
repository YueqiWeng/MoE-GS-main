# Component ablation — all scenes

CSV: [ablations_all_scenes.csv](ablations_all_scenes.csv)

## Completion

| Variant | Done |
|---|---:|
| w/o MoE | 19/19 |
| w/o tree | 19/19 |
| w/o CL | 19/19 |
| w/o L_load | 19/19 |
| w/o warm-start | 19/19 |
| Full | 19/19 |

## Mean PSNR by dataset

| Dataset | w/o MoE | w/o tree | w/o CL | w/o L_load | w/o warm-start | Full |
|---|---:|---:|---:|---:|---:|---:|
| mipnerf360 | 29.187 | 29.421 | 29.795 | 29.805 | 29.453 | 29.796 |
| tandt | 24.117 | 24.365 | 24.561 | 24.532 | 24.275 | 24.523 |
| db | 30.236 | 30.378 | 30.441 | 30.505 | 30.141 | 30.500 |
| nerfsynthetic | 33.452 | 33.259 | 33.563 | 33.555 | 33.195 | 33.573 |
| **all** | 30.559 | 30.606 | 30.899 | 30.903 | 30.556 | 30.906 |

## Run missing component ablations

```bash
bash sweep.sh --suite ablate-components --ablate-components nocl,noload,nowarm --skip-done -g 0
```

Regenerate this table:

```bash
python scripts/collect_ablation_table.py
```
