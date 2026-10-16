# Ablation study plan (no GNN)

This document is the plan for the **component ablation** of our method.  
It mirrors Scene Experts §4.4 (Table 4), adapted to **Growing Tree MoE** without GNN.

Hyperparameter analysis (\(N\), Top-\(k\), grow count/schedule) is **separate** and not part of this plan.

---

## 1. Goal

Answer: **which pieces of the full method are necessary for the reported quality?**

We start from a fixed **Full** model and remove **one** component at a time.  
If PSNR/SSIM/LPIPS drop, that component is justified in the paper.

---

## 2. Full model (reference)

**Name:** `moe_tree`  
**Claim:** Growing Tree MoE on Scaffold-GS anchors, no GNN.

| Knob | Full setting |
|---|---|
| Architecture | `moe_arch=dynamic_tree` |
| Start width | 8 leaves (= flat MoE, 1:1) |
| Growth | up to 4 splits (`moe_max_total_grows=4`), checks at 6k / 10k / 14k / 18k |
| Routing | Top-2 after curriculum |
| Curriculum (CL) | dense softmax for first **3000** iters (`moe_warmup=3000`), then Top-2 |
| \(L_\text{load}\) | \(\lambda=0.005\) |
| \(L_\text{opacity}\) | \(\lambda=0.01\) |
| Router entropy | \(\lambda=0.002\) |
| Warm-start | finished flat `moe` MLP + point cloud; keep MoE points (`update_until=0`) |
| Training | 30k iterations |
| GNN | **off** |

Output folder: `outputs/<dataset>/<scene>/moe_tree/<timestamp>/`

---

## 3. Evaluation protocol

### Scenes

**Full ablation: all 19 scenes** (same coverage as grow ablation):

| Suite | Scenes |
|---|---|
| mipnerf360 | bicycle, bonsai, counter, garden, kitchen, room, stump |
| Tanks&Temples | train, truck |
| Deep Blending | drjohnson, playroom |
| NeRF-Synthetic | chair, drums, ficus, hotdog, lego, materials, mic, ship |

Report **per-dataset mean** and **overall mean** (PSNR / SSIM / LPIPS). Per-scene table in appendix.

Results table: [ablations_all_scenes.md](ablations_all_scenes.md) · CSV: [ablations_all_scenes.csv](ablations_all_scenes.csv)

### Controls (must stay fixed across rows)

- Same scene voxel / `update_init_factor` / lod (from `sweep.sh` hparams)
- Same iteration budget (30k)
- Same random protocol as usual training
- For all **warm-started** tree rows: same flat-MoE checkpoint per scene
- **One change only** per ablation row

### Metrics

| Metric | Role |
|---|---|
| PSNR ↑ | primary |
| SSIM ↑ | secondary |
| LPIPS ↓ | secondary |
| size (MB) | optional (paper reports it; add if easy) |

---

## 4. Ablation variants (Table A)

| # | Table name | Code / folder | One-sentence meaning |
|---|---|---|---|
| 1 | w/o MoE | `baseline` | Shared MLP; no experts / router / tree |
| 2 | w/o tree | `moe` | Flat MoE only; no hierarchy or growth |
| 3 | w/o CL | `ablate_nocl` | Full tree, but Top-2 from iter 0 (no dense warmup) |
| 4 | w/o \(L_\text{load}\) | `ablate_noload` | Full tree, load-balancing loss off |
| 5 | w/o warm-start | `ablate_nowarm` | Full tree recipe, no flat-MoE init |
| 6 | **Full** | `moe_tree` | All components on |

Optional (only if opacity loss is a paper claim):

| # | Table name | Folder | Change |
|---|---|---|---|
| 7 | w/o \(L_\text{opacity}\) | `ablate_noopac` | `lambda_opacity=0` |

### Detailed variant specs

#### (1) w/o MoE — `baseline`

- **Remove:** MoE experts, router, tree, MoE auxiliary losses, MoE warm-start  
- **Keep:** Scaffold-GS geometry + shared MLP  
- **Answers:** Is multi-expert modeling needed at all?  
- **Status:** reuse existing runs  
- **Paper analog:** Scene Experts “w/o MoE”

#### (2) w/o tree — `moe`

- **Remove:** tree structure and leaf growth  
- **Keep:** 8 flat experts, Top-2, CL (3k), \(L_\text{load}\), \(L_\text{opacity}\), entropy  
- **Train:** from scratch as flat MoE (this run is also the warm-start source for Full)  
- **Answers:** Does hierarchy/growth help beyond flat MoE?  
- **Status:** reuse existing runs  
- **Paper analog:** none (our addition; they have no tree)

#### (3) w/o CL — `ablate_nocl`

- **Change:** `moe_warmup=0` only  
- **Keep:** tree, warm-start, grows=4, \(L_\text{load}\), Top-2 immediately  
- **Answers:** Is dense→sparse curriculum necessary?  
- **Status:** to run (sweep `nocl`)  
- **Paper analog:** Scene Experts “w/o CL”

#### (4) w/o \(L_\text{load}\) — `ablate_noload`

- **Change:** `lambda_load=0` only  
- **Keep:** tree, warm-start, CL, grows  
- **Answers:** Does load balancing help quality / expert use?  
- **Status:** to run (sweep `noload`)  
- **Paper analog:** Scene Experts “w/o \(L_\text{load}\)”

#### (5) w/o warm-start — `ablate_nowarm`

- **Change:** `--no_warm_start` (no MLP/ply from flat MoE; normal densification)  
- **Keep:** tree arch, CL, grows, losses  
- **Answers:** How much of the tree gain is warm-start + continued training vs the architecture itself?  
- **Status:** to run (sweep `nowarm`)  
- **Paper analog:** none (our addition; motivated by grow ablation where grows=0 already gains a lot)

#### (6) Full — `moe_tree`

- All components enabled (Section 2)  
- **Status:** reuse existing runs  

### Cheat sheet

| Variant | MoE | Tree | Warm-start | CL | \(L_\text{load}\) |
|---|---|---|---|---|---|
| w/o MoE | ✗ | ✗ | ✗ | — | — |
| w/o tree | flat | ✗ | ✗* | ✓ | ✓ |
| w/o CL | tree | ✓ | ✓ | ✗ | ✓ |
| w/o \(L_\text{load}\) | tree | ✓ | ✓ | ✓ | ✗ |
| w/o warm-start | tree | ✓ | ✗ | ✓ | ✓ |
| **Full** | tree | ✓ | ✓ | ✓ | ✓ |

\* Flat MoE is trained from scratch; it *provides* warm-start for tree rows.

---

## 5. What this plan does *not* include

| Out of scope | Reason |
|---|---|
| GNN / `moe_gnn` / `tree_gnn` | Not part of the claimed full model |
| Grow count 0/2/4 | Hyperparameter study (already done; see `routing/ablate_tree_grows.md`) |
| \(N\) / Top-\(k\) sweep | Hyperparameter study (after this table) |
| Grow schedule / which head grows | Optional follow-up, not needed for Table A |

---

## 6. Compute plan

| Item | Count |
|---|---:|
| Scenes | **19** |
| New variants | 3 (`nocl`, `noload`, `nowarm`) |
| **New runs** | **57** (19×3); **45** remaining after DB+T&T pilot |
| Reused runs | `baseline`, `moe`, `moe_tree` × 19 scenes |

Rough time: ~30–50 min/run → about **1–1.5 days** on one GPU for remaining 45 runs.

### Run commands

```bash
# full 19-scene sweep (skips finished)
bash sweep.sh --suite ablate-components --ablate-components nocl,noload,nowarm --skip-done -g 0

# pilot only (4 scenes: DB + T&T)
bash sweep.sh --suite ablate-components-quick --ablate-components nocl,noload,nowarm --skip-done -g 0

# background
nohup bash sweep.sh --suite ablate-components --ablate-components nocl,noload,nowarm \
  --skip-done -g 0 > sweep_logs/ablate_components_all19.out 2>&1 &

# refresh results table
python scripts/collect_ablation_table.py
```

Smoke test (one scene):

```bash
bash train_method.sh --method moe_tree -d db/playroom -g 0 -l ablate_nocl --moe_warmup 0
bash train_method.sh --method moe_tree -d db/playroom -g 0 -l ablate_noload --lambda_load 0
bash train_method.sh --method moe_tree -d db/playroom -g 0 -l ablate_nowarm --no_warm_start
```

Optional opacity row:

```bash
bash sweep.sh --suite ablate-components --ablate-components noopac --skip-done -g 0
```

---

## 7. Expected outcomes (hypotheses)

| Variant | Expected vs Full |
|---|---|
| w/o MoE | **Largest** drop |
| w/o warm-start | **Large** drop (warm-start carries much of the tree gain) |
| w/o tree | Medium drop |
| w/o CL | Clear but smaller drop |
| w/o \(L_\text{load}\) | Small–medium drop; routing more collapsed |
| Full | Best (or tied) mean PSNR |

If w/o tree ≈ Full, hierarchy is weak on these scenes — still report honestly and lean on warm-start / routing analysis.

---

## 8. Draft numbers already in hand (reuse rows)

Dataset means (latest runs):

| Ablation | DeepBlending PSNR | Tanks&Temples PSNR |
|---|---:|---:|
| w/o MoE | 30.236 | 24.117 |
| w/o tree | 30.378 | 24.365 |
| Full | **30.500** | **24.523** |

Per-scene PSNR:

| Scene | w/o MoE | w/o tree | Full |
|---|---:|---:|---:|
| drjohnson | 29.643 | 29.736 | 29.861 |
| playroom | 30.829 | 31.021 | 31.139 |
| train | 22.338 | 22.695 | 22.745 |
| truck | 25.896 | 26.035 | 26.301 |

After the 12 new runs, extend this table with w/o CL, w/o \(L_\text{load}\), w/o warm-start.

---

## 9. Deliverables when finished

1. **Table A** in paper style (dataset columns × ablation rows)  
2. Updated `documents/method_summary/ablations.md` with final numbers  
3. One paragraph: each component’s contribution (drop size)  
4. Optional: routing CV / util plot for Full vs w/o \(L_\text{load}\) (like Scene Experts Fig. 3)

---

## 10. Execution checklist

- [x] Define Full model and variants (no GNN)
- [x] Wire CLI flags in `train_method.sh`
- [x] Wire `sweep.sh --suite ablate-components --ablate-components …`
- [x] Reuse baseline / moe / moe_tree numbers for 4 scenes
- [ ] Run `nocl,noload,nowarm` (12 jobs)
- [ ] Fill final Table A (PSNR / SSIM / LPIPS)
- [ ] Write short takeaway paragraph
- [ ] (Later) hyperparams: \(N\), Top-\(k\); keep grow figure as-is
