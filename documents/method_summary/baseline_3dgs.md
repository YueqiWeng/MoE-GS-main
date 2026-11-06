# Baseline: official 3D Gaussian Splatting

Paper: Kerbl, Kopanas, Leimkühler, Drettakis. **3D Gaussian Splatting for Real-Time Radiance Field Rendering**, ACM TOG 2023 (`kerbl20233d`).

This is **not** Scaffold-GS. `baseline` in this repo is still shared-MLP Scaffold-GS. `3dgs` clones [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) into `third_party/gaussian-splatting` and trains that code on the same `data/<suite>/<scene>` trees.

## Protocol (match existing `baseline` / `moe` / `moe_tree` runs)

Do **not** use the 3D-GS paper `images_4` / `images_2` recipe. Existing runs in this project all used:

| Flag | Value | Why |
|---|---|---|
| `--images` | `images` | same folder as existing `cfg_args` |
| `--resolution` / `-r` | default `-1` | cap width at 1600 (bicycle GT is 1600×1063, not 1237×822) |
| `--eval` | on | every 8th COLMAP camera; Blender `transforms_test.json` |
| `--white_background` | off | existing NeRF-Synthetic `cfg_args` have `white_background=False` |
| BungeeNeRF split | `lod=30` | first 31 cameras after sort (31 test views on hollywood) |
| BungeeNeRF extra | `--data_device cpu --densify_grad_threshold 0.0004` | official 3D-GS OOM'd at default densify on 16GB |
| Blender loader | `np.uint8` + skip missing frames | stock `np.byte` crashes Pillow; hotdog is missing `r_99.png` |
| Blender composite | RGBA onto black in `loadCam` | stock 3D-GS reloads PNG and trains only on alpha, background explodes |
| iterations | 30k | same as other methods |
| metrics | this repo `metrics.py` | PSNR / SSIM / LPIPS-VGG → `ours_30000` |

Metrics land in `outputs/<scene>/3dgs/<timestamp>/results.json` so `sweep.sh` treats 3D-GS like the other methods.

## Run all project benchmarks

```bash
# 19-scene paper split (mip360 + tandt + db + nerfsynthetic)
bash sweep.sh --suite paper --methods 3dgs --skip-done -g 0

# every suite this repo knows (adds BungeeNeRF)
bash sweep.sh --suite all --methods 3dgs --skip-done -g 0
```

Single scene:

```bash
bash train_method.sh --method 3dgs -d mipnerf360/kitchen -g 0
```

## Results (all 27 scenes on disk)

Same protocol as `baseline` / `moe` / `moe_tree`. `blending` is a symlink to `db`. Mip-NeRF 360 `flowers` / `treehill` are not present. CSV: `results_3dgs_all.csv`.

| Scene | 3D-GS | Scaffold | MoE | Tree |
|---|---:|---:|---:|---:|
| bicycle | 25.17 | 25.14 | 25.27 | 25.47 |
| bonsai | 32.50 | 32.52 | 33.07 | 33.75 |
| counter | 29.18 | 29.47 | 29.66 | 30.09 |
| garden | 27.37 | 27.27 | 27.44 | 27.68 |
| kitchen | 31.22 | 31.24 | 31.71 | 32.24 |
| room | 31.84 | 32.10 | 32.14 | 32.56 |
| stump | 26.65 | 26.56 | 26.66 | 26.79 |
| *Mip-NeRF 360 mean* | 29.13 | 29.19 | 29.42 | 29.80 |
| train | 21.94 | 22.34 | 22.70 | 22.75 |
| truck | 25.47 | 25.90 | 26.04 | 26.30 |
| *Tanks & Temples mean* | 23.71 | 24.12 | 24.37 | 24.52 |
| drjohnson | 29.49 | 29.64 | 29.74 | 29.86 |
| playroom | 30.09 | 30.83 | 31.02 | 31.14 |
| *Deep Blending mean* | 29.79 | 30.24 | 30.38 | 30.50 |
| chair | 35.73 | 35.01 | 34.90 | 35.19 |
| drums | 26.29 | 26.22 | 26.16 | 26.33 |
| ficus | 35.55 | 34.90 | 34.20 | 34.60 |
| hotdog | 38.12 | 37.85 | 37.74 | 38.09 |
| lego | 36.07 | 35.37 | 35.52 | 35.86 |
| materials | 30.53 | 30.37 | 30.17 | 30.53 |
| mic | 36.89 | 36.50 | 36.12 | 36.58 |
| ship | 31.72 | 31.41 | 31.26 | 31.40 |
| *NeRF-Synthetic mean* | 33.86 | 33.45 | 33.26 | 33.57 |
| amsterdam | 25.44 | 26.82 | 27.04 | 27.12 |
| bilbao | 26.77 | 28.06 | 28.17 | 28.09 |
| hollywood | 22.83 | 23.84 | 23.74 | 23.14 |
| pompidou | 20.99 | 19.61 | 19.63 | 19.29 |
| quebec | 28.36 | 28.38 | 28.55 | 28.26 |
| rome | 22.45 | 23.40 | 23.11 | 21.87 |
| barcelona | 26.03 | 26.69 | 26.91 | 26.87 |
| chicago | 22.35 | 21.52 | 21.27 | 20.45 |
| *BungeeNeRF mean* | 24.40 | 24.79 | 24.80 | 24.38 |
| **paper 19 mean** | **30.62** | **30.56** | **30.61** | **30.91** |
| **all 27 mean** | **28.78** | **28.85** | **28.89** | **28.97** |

Paper-19 SSIM / LPIPS: 3D-GS 0.9144 / 0.1233, Scaffold 0.9124 / 0.1305, MoE 0.9143 / 0.1244, Tree 0.9167 / 0.1206.
