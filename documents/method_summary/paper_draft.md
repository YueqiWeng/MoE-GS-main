# Growing Tree Mixture-of-Experts for Scaffold-Gaussian Splatting

**Draft for CVPR** — based on completed experiments (19 scenes; component ablation; grow-count ablation).  
**Claimed full model:** Growing Tree MoE on Scaffold-GS anchors (**no GNN**), warm-started from flat MoE.

---

## Abstract (draft)

Anchor-based 3D Gaussian Splatting (Scaffold-GS) decodes neural Gaussian attributes with shared MLPs, which underfits spatially heterogeneous appearance. We replace each attribute decoder with a **growing tree mixture-of-experts (Tree MoE)** that (i) routes anchors to specialized experts, (ii) **warm-starts** from a trained flat MoE, and (iii) optionally **splits** high-difficulty leaves under an information-theoretic growth rule. On 19 scenes spanning Mip-NeRF 360, Tanks & Temples, Deep Blending, and NeRF-Synthetic, Tree MoE improves mean PSNR by **+0.35 dB** over Scaffold-GS and **+0.30 dB** over flat MoE. Ablations show that **MoE capacity** and **warm-start** are essential, while curriculum and load balancing contribute little to PSNR; most of the gain already appears with **zero leaf splits**, with further growth adding only marginal quality.

---

## 1. Introduction

### 1.1 Background

Novel-view synthesis from multi-view images has progressed from continuous neural fields (NeRF) to explicit **3D Gaussian Splatting (3DGS)**, which represents a scene as anisotropic Gaussians rasterized in real time. **Scaffold-GS** further organizes the representation around a sparse set of *anchors*: each anchor \(i\) stores a position \(\mathbf{x}_i\in\mathbb{R}^3\) and a learnable feature \(\mathbf{f}_i\in\mathbb{R}^{C}\). View-conditioned features \(\mathbf{u}_i\) are decoded by MLPs into opacity, covariance, and color of neural Gaussians that are then α-blended.

Formally, for a camera \(\mathcal{C}\), Scaffold-GS predicts an image
\[
\hat{I}(\mathcal{C})
=
\mathcal{R}\!\left(
\big\{\,(\alpha_j,\boldsymbol{\Sigma}_j,\mathbf{c}_j)\,
\big\}_{j=1}^{M}
\right),
\]
where \(\mathcal{R}\) denotes differentiable Gaussian rasterization and the attributes \((\alpha_j,\boldsymbol{\Sigma}_j,\mathbf{c}_j)\) are produced by attribute networks \(F_\alpha,F_\Sigma,F_c\) applied to view features of visible anchors.

### 1.2 Motivation

A single shared MLP per attribute imposes a **uniform capacity** across the scene. Real scenes mix easy regions (flat walls, sky) with hard ones (specular highlights, fine foliage, thin structures). Mixture-of-Experts (MoE) methods allocate different experts to different inputs, but a **fixed flat bank of \(E\) experts** still forces a global capacity budget: easy regions waste experts, while hard regions remain under-parameterized.

This suggests two complementary needs:

1. **Specialization:** route anchors to different experts based on local features.
2. **Adaptive capacity:** grow additional experts only where residual difficulty remains high.

### 1.3 Purpose and contributions

We propose a **Growing Tree MoE** decoder for Scaffold-GS with the following goals:

- Improve reconstruction quality over shared-MLP Scaffold-GS and flat MoE.
- Allocate capacity hierarchically via leaf splits, guided by traffic mass, residual loss, and feature heterogeneity.
- Provide a practical two-stage recipe: train flat MoE → warm-start tree → optional growth.

**Contributions.**

1. A **hierarchical Tree MoE** attribute decoder (opacity / covariance / color) with Top-\(k\) routing and Switch-style regularization.
2. An **information-guided growth** procedure that splits leaves only under capacity–difficulty mismatch.
3. A **warm-start training algorithm** from flat MoE that accounts for most of the measured gains.
4. A comprehensive study on **19 scenes** with component and grow-count ablations.

---

## 2. Related work (brief placeholders)

- **3DGS / Scaffold-GS:** explicit Gaussian primitives; anchor-based neural Gaussians.
- **MoE for graphics / radiance fields:** Scene Experts and related sparse expert routing.
- **Adaptive / hierarchical networks:** progressive growing, tree-structured routing, capacity allocation.

*(Expand with citations in the final submission.)*

---

## 3. Method

### 3.1 Preliminaries: Scaffold-GS backbone

Let \(\mathcal{A}=\{(\mathbf{x}_i,\mathbf{f}_i)\}_{i=1}^{N}\) be anchors. For a viewpoint, visible anchors yield view features \(\mathbf{u}_i=\phi(\mathbf{f}_i,\mathbf{d}_i,\ldots)\), where \(\mathbf{d}_i\) is the view direction. Attribute networks map \(\mathbf{u}_i\) to neural-Gaussian parameters. We keep Scaffold-GS geometry densification / pruning unchanged unless noted; for warm-started tree runs we freeze densification (`update_until=0`) and continue from the flat-MoE point cloud.

### 3.2 Flat Mixture-of-Experts decoder

For each attribute head \(h\in\{\alpha,\Sigma,c\}\), we replace the shared MLP by \(E\) experts \(\{f_e^{(h)}\}_{e=1}^{E}\) and a router \(r^{(h)}\).

Given router features \(\mathbf{z}_i=\mathbf{f}_i\) (anchor feature only; **no GNN** in the claimed model),
\[
\mathbf{g}_i
=
\mathrm{softmax}\!\big(r^{(h)}(\mathbf{z}_i)\big)
\in\Delta^{E-1}.
\]

**Curriculum (CL).** For the first \(T_{\mathrm{w}}\) iterations (default \(T_{\mathrm{w}}=3000\)), mix with dense gates:
\[
\mathbf{w}_i=\mathbf{g}_i.
\]
Afterwards, apply Top-\(k\) sparsification (\(k=2\)):
\[
\mathbf{w}_i
=
\frac{\mathrm{TopK}(\mathbf{g}_i,k)}
{\|\mathrm{TopK}(\mathbf{g}_i,k)\|_1}.
\]

The expert output is
\[
F_h(\mathbf{u}_i)
=
\sum_{e=1}^{E} w_{i,e}\, f_e^{(h)}(\mathbf{u}_i).
\]

In our flat MoE baseline: \(E=8\), \(k=2\).

### 3.3 Growing Tree MoE

#### 3.3.1 Tree structure

Each attribute head maintains a rooted tree. Internal nodes hold routers; leaves hold experts. At initialization of the tree stage, we place **\(B=8\) leaves under the root** in 1:1 correspondence with the flat MoE experts (warm-start compatible).

A path product defines the probability of reaching leaf \(\ell\):
\[
\pi_i(\ell)
=
\prod_{(\nu\to c)\in\mathrm{path}(\ell)}
w_{i,\nu\to c},
\]
and
\[
F_h(\mathbf{u}_i)
=
\sum_{\ell\in\mathcal{L}}
\pi_i(\ell)\, f_\ell^{(h)}(\mathbf{u}_i).
\]

When sparse Top-\(k\) is active at a node with \(b\) children, only \(k=\min(2,b)\) children receive mass (renormalized). Softmax path probabilities \(\tilde{\pi}_i(\ell)\) are retained for auxiliary losses and statistics.

#### 3.3.2 Leaf split (growth)

A grow event replaces a leaf \(\ell\) of depth \(d\) by an internal router with \(G=2\) children (binary grow by default), cloning the parent expert with small Gaussian noise \(\varepsilon\sim\mathcal{N}(0,\sigma^2 I)\):
\[
f_{\ell\to c}^{(h)}
\leftarrow
f_\ell^{(h)}+\varepsilon_c,
\quad c\in\{0,1\}.
\]
Constraints: tree depth \(\le D_{\max}=3\), total grows \(\le G_{\max}\) (canonical \(G_{\max}=4\)), leaf count \(\le L_{\max}=14\). Per check, at most \(H_{\mathrm{head}}=2\) heads grow and at most \(1\) leaf per head.

### 3.4 Information-guided growth score

Every leaf has the **same parameter count**, hence roughly fixed capacity \(C\). We treat residual difficulty as conditional entropy \(H(Y\mid\ell)\) and use training proxies.

Over a window between grow checks, for leaf \(\ell\):

- **Traffic mass:**
  \[
  m_\ell=\sum_{t}\sum_{i}\pi_i^{(t)}(\ell),\qquad
  p(\ell)=\frac{m_\ell}{\sum_{\ell'}m_{\ell'}}.
  \]
- **Mass-weighted loss** (proxy for \(H(Y\mid\ell)\)):
  \[
  L_\ell
  =
  \frac{\sum_{t} m_\ell^{(t)}\, \mathcal{L}_1^{(t)}}
  {\sum_{t} m_\ell^{(t)}}.
  \]
- **Feature heterogeneity:**
  \[
  v_\ell
  =
  \mathbb{E}\!\big[\mathrm{Var}_{\pi(\cdot\mid\ell)}(\mathbf{f})\big].
  \]

Relative difficulty and score:
\[
R_\ell=\frac{L_\ell}{\mathrm{median}_{\ell'}(L_{\ell'})},
\quad
\mathrm{excess}_\ell=\max(0,R_\ell-1),
\]
\[
S_\ell
=
p(\ell)\,(1+\mathrm{excess}_\ell)^2\,(1+0.75\,v_\ell).
\]

**Scene budget** from heterogeneity \(H=L_{\max}/\mathrm{median}(L)\):
\[
\mathrm{budget}(H)
=
\begin{cases}
0,& H < \kappa_{\mathrm{easy}},\\
1,& \kappa_{\mathrm{easy}}\le H < \kappa_{\mathrm{hard}},\\
G_{\max},& \text{otherwise}.
\end{cases}
\]

Grow leaf \(\ell\) only if remaining budget \(>0\) and all gates hold (defaults used in experiments):
\[
\begin{aligned}
p(\ell)&\ge p_{\min},\\
R_\ell&\ge \rho_L,\\
v_\ell&\ge \mathrm{median}(v),\\
S_\ell&\ge \rho_S\cdot\mathrm{median}(S).
\end{aligned}
\]

After a successful split, routing is temporarily dense for \(T_{\mathrm{dense}}\) iterations so new children can attract mass, then Top-\(k\) resumes.

### 3.5 Training objectives

**Photometric + geometry (Scaffold-GS):**
\[
\mathcal{L}_{\mathrm{rgb}}
=
(1-\lambda_{\mathrm{SSIM}})\,\mathcal{L}_1(\hat{I},I)
+
\lambda_{\mathrm{SSIM}}\,\mathcal{L}_{\mathrm{SSIM}}(\hat{I},I)
+
\lambda_{\mathrm{s}}\,\mathbb{E}[\textstyle\prod_d s_d].
\]

**MoE auxiliaries** (after / during curriculum as scheduled):

1. **Opacity regularizer**
   \[
   \mathcal{L}_{\mathrm{op}}
   =
   \mathbb{E}\big[\mathrm{ReLU}(\tau-\alpha)\big].
   \]

2. **Switch load balancing** over routers / soft gates \(P=\mathbb{E}_i[\mathbf{g}_i]\), fraction \(f\):
   \[
   \mathcal{L}_{\mathrm{load}}
   =
   E\,\langle P,f\rangle.
   \]

3. **Router entropy** (encourages concentration under sparse routing):
   \[
   \mathcal{L}_{\mathrm{ent}}
   =
   -\mathbb{E}_i\big[\tilde{\boldsymbol{\pi}}_i^\top\log\tilde{\boldsymbol{\pi}}_i\big].
   \]

**Total loss:**
\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{rgb}}
+
\lambda_{\mathrm{op}}\mathcal{L}_{\mathrm{op}}
+
\lambda_{\mathrm{load}}(t)\,\mathcal{L}_{\mathrm{load}}
+
\lambda_{\mathrm{ent}}(t)\,\mathcal{L}_{\mathrm{ent}},
\]
with schedule
\[
\lambda_{\mathrm{load}}(t)=
\begin{cases}
\lambda_{\mathrm{load}},& t\le T_{\mathrm{w}},\\
0.1\,\lambda_{\mathrm{load}},& t>T_{\mathrm{w}},
\end{cases}
\quad
\lambda_{\mathrm{ent}}(t)=
\begin{cases}
0,& t\le T_{\mathrm{w}},\\
\lambda_{\mathrm{ent}},& t>T_{\mathrm{w}}.
\end{cases}
\]

Canonical coefficients: \(\lambda_{\mathrm{op}}=0.01\), \(\lambda_{\mathrm{load}}=0.005\), \(\lambda_{\mathrm{ent}}=0.002\).

### 3.6 Two-stage training recipe

**Stage A — Flat MoE.** Train Scaffold-GS + flat MoE (\(E=8\)) for \(T=30{,}000\) iterations from SfM initialization (with densification as in Scaffold-GS).

**Stage B — Tree MoE (Full).** Initialize tree leaves and root router from Stage A checkpoints (experts + router; optional Gaussian ply). Train another \(30{,}000\) iterations with densification off. Grow checks at
\[
t\in\{6000,10000,14000,18000\}
\quad\text{within }[t_{\mathrm{start}},t_{\mathrm{stop}}]=[6000,20000].
\]

---

## 4. Training algorithm

### Algorithm 1 — Flat MoE training (Stage A)

**Input:** multi-view images, SfM points; \(E=8\), \(k=2\), \(T_{\mathrm{w}}=3000\), \(T=30000\).  
**Output:** checkpoint \(\Theta_{\mathrm{flat}}\) (MLPs + anchors / Gaussians).

1. Initialize Scaffold-GS anchors from SfM; initialize \(E\) experts and routers per attribute.
2. **For** \(t=1,\ldots,T\):
   1. Sample camera \(\mathcal{C}\); render \(\hat{I}\).
   2. Set sparse \(\leftarrow\) (\(t>T_{\mathrm{w}}\)).
   3. Compute \(\mathcal{L}\) as in Sec. 3.5; backpropagate; Adam step.
   4. Update anchors / densify while \(t < t_{\mathrm{densify}}\).
3. **Return** \(\Theta_{\mathrm{flat}}\).

### Algorithm 2 — Growing Tree MoE training (Stage B)

**Input:** \(\Theta_{\mathrm{flat}}\); \(G_{\max}\), grow schedule, growth gates.  
**Output:** tree MoE model \(\Theta_{\mathrm{tree}}\).

1. Build tree with \(B=8\) root leaves; **copy** flat experts / routers into leaves (1:1).
2. Load Gaussian / anchor state from \(\Theta_{\mathrm{flat}}\); set densification off.
3. Reset leaf statistics \(\{m_\ell,L_\ell,v_\ell\}\leftarrow 0\).
4. **For** \(t=1,\ldots,T\):
   1. Sample camera; render with tree routing (dense if \(t\le T_{\mathrm{w}}\) or inside post-grow dense window).
   2. Compute \(\mathcal{L}\); backward; optimizer step.
   3. `record_leaf_stats` using \(\mathcal{L}_1\) and path masses \(\pi_i(\ell)\).
   4. **If** \(t\) is a grow check and \(t\in[t_{\mathrm{start}},t_{\mathrm{stop}}]\):
      - For each head (up to \(H_{\mathrm{head}}\) heads): select top candidate by \(S_\ell\) subject to gates; call `grow_leaf`.
      - If any grow succeeded: enable dense routing for \(T_{\mathrm{dense}}\) iters; reset stats for split leaves.
5. **Return** \(\Theta_{\mathrm{tree}}\).

### Algorithm 3 — `grow_leaf`(\(\ell\))

1. If depth(\(\ell\)) \(\ge D_{\max}\) or \(|\mathrm{history}|\ge G_{\max}\) or leaf budget exceeded: **abort**.
2. Create router \(r_{\ell}\) with \(G=2\) outputs; clone expert \(\to\) children \(\ell_{c0},\ell_{c1}\) (+ noise).
3. Replace leaf \(\ell\) by internal node \((r_{\ell},\{\ell_{c0},\ell_{c1}\})\).
4. Append \(\ell\) to growth history; clear stats of \(\ell\).

---

## 5. Experiments

### 5.1 Setup

**Datasets (19 scenes):**

| Suite | Scenes |
|---|---|
| Mip-NeRF 360 | bicycle, bonsai, counter, garden, kitchen, room, stump |
| Tanks & Temples | train, truck |
| Deep Blending | drjohnson, playroom |
| NeRF-Synthetic | chair, drums, ficus, hotdog, lego, materials, mic, ship |

**Metrics:** PSNR↑, SSIM↑, LPIPS↓ on held-out views (`ours_30000`).  
**Compared rows:** Scaffold-GS (w/o MoE), flat MoE (w/o tree), Full Tree MoE; plus ablations w/o CL, w/o \(L_{\mathrm{load}}\), w/o warm-start; grow counts \(G_{\max}\in\{0,2,4\}\).

### 5.2 Main quantitative results

**Overall mean (19 scenes):**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Scaffold-GS (w/o MoE) | 30.559 | 0.9124 | 0.1305 |
| Flat MoE (w/o tree) | 30.606 | 0.9143 | 0.1244 |
| **Tree MoE (Full)** | **30.906** | **0.9167** | **0.1206** |

**Δ vs Scaffold-GS:** \(+0.35\) dB PSNR.  
**Δ vs flat MoE:** \(+0.30\) dB PSNR; Full wins on **19/19** scenes vs flat MoE.

**Per-dataset PSNR:**

| Dataset | Scaffold-GS | Flat MoE | Tree MoE |
|---|---:|---:|---:|
| Mip-NeRF 360 | 29.187 | 29.421 | **29.796** |
| Tanks & Temples | 24.117 | 24.365 | **24.523** |
| Deep Blending | 30.236 | 30.378 | **30.500** |
| NeRF-Synthetic | 33.452 | 33.259 | **33.573** |

Largest absolute gains vs flat MoE appear on hard real scenes (e.g. bonsai \(+0.68\), kitchen \(+0.53\), counter \(+0.43\) dB).

### 5.3 Component ablation

One-at-a-time removal from Full (19/19 complete):

| Variant | PSNR | Δ vs Full |
|---|---:|---:|
| w/o MoE | 30.559 | −0.35 |
| w/o tree | 30.606 | −0.30 |
| w/o CL | 30.899 | −0.01 |
| w/o \(L_{\mathrm{load}}\) | 30.903 | −0.00 |
| w/o warm-start | 30.556 | −0.35 |
| **Full** | **30.906** | — |

**Findings.**

- **Necessary for quality:** MoE experts and warm-start (each ≈ −0.35 dB when removed).
- **Hierarchy / continued tree training:** Flat MoE lags Full by −0.30 dB on average.
- **CL and \(L_{\mathrm{load}}\):** negligible PSNR effect (within noise); may still matter for routing stability.

### 5.4 Grow-count ablation (\(G_{\max}\in\{0,2,4\}\))

| Setting | Mean PSNR | Δ vs flat MoE |
|---|---:|---:|
| Flat MoE | 30.606 | — |
| grows=0 | 30.908 | +0.302 |
| grows=2 | 30.902 | +0.297 |
| grows=4 (Full recipe) | **30.913** | **+0.308** |

Almost all improvement over flat MoE is already present at **grows=0** (warm-start + tree training recipe without splits). Extra splits change mean PSNR by only ~\(10^{-2}\) dB. Scene-wise, grows={0,2,4} each win on a subset (5 / 7 / 7 scenes). Synthetic scenes benefit slightly more from growth (e.g. lego \(+0.14\) dB from g0→g4).

### 5.5 Efficiency (summary)

Approximate wall-clock (mean over measured runs): baseline \(16\) min, flat MoE \(31\) min, Tree MoE \(41\) min (~**2.6×** baseline). Peak GPU memory remains comparable to flat MoE (~4 GB class on representative scenes).

### 5.6 Qualitative / routing

Routing dumps show color-head trees evolving from 8 flat leaves to 10–11 leaves under grows=2/4, with mass concentrating on children of split nodes (e.g. kitchen grows=2: splits of \(L6\_7\) then \(L0\_1\)). Sample gate entropy stays low under Top-2; utilization CV rises as parents split.

---

## 6. Discussion

### 6.1 What drives the gains?

Empirically, the dominant factors are:

1. **Expert specialization (MoE)** over a shared MLP.
2. **Warm-start + continued optimization** in the tree recipe — even with \(G_{\max}=0\).

Dynamic splits are a **secondary, scene-dependent** refinement: they implement the capacity–difficulty story cleanly, but average PSNR gains are small under the current schedule and gates. This is consistent with the information-theoretic view: many leaves are already near capacity after Stage A, so verified \(\Delta I\) from further splits is limited.

### 6.2 Implications for the narrative

For CVPR, a honest framing is:

> We introduce Tree MoE for Scaffold-GS with a warm-started hierarchical decoder and an optional information-guided growth rule. Growth provides adaptive capacity and interpretable specialization; **warm-started MoE training** is the primary driver of quantitative gains, while growth yields modest additional improvements on selected scenes.

Avoid over-claiming that “more grows ⇒ always better”; the grow ablation contradicts that.

### 6.3 Limitations

- Two-stage training increases total compute (~2.6× vs baseline for Stage B alone, plus Stage A).
- Growth hyperparameters (\(p_{\min}\), ratios, schedule) are manually set; per-scene optimal \(G_{\max}\) varies.
- GNN dual-routing was explored in pilot work but is **not** part of the claimed Full model here.
- CL / load loss do not move PSNR much; stronger justification needs routing / utilization metrics in the paper body.

### 6.4 Future work

- Learned grow policies / differentiable capacity allocation.
- Joint single-stage training without a separate flat MoE stage.
- Stronger mutual-information estimators for \(\Delta I\) after candidate splits.
- Hyperparameter study of \(E\) and Top-\(k\).

---

## 7. Conclusion

We presented a **Growing Tree Mixture-of-Experts** decoder for Scaffold-Gaussian Splatting. By combining Top-\(k\) expert routing, Switch-style regularization, warm-start from flat MoE, and information-guided leaf growth, the method improves novel-view quality across 19 standard scenes (**+0.35 dB** vs Scaffold-GS, **+0.30 dB** vs flat MoE). Ablations identify **MoE capacity** and **warm-start** as essential components; curriculum and load balancing are nearly PSNR-neutral; and **most of the tree-stage gain appears without any leaf splits**, with growth acting as a fine-grained, scene-dependent capacity expansion. These results support hierarchical experts as a practical path to adaptive capacity in anchor-based 3DGS, while clarifying which ingredients truly matter for reconstruction quality.

---

## Appendix A — Hyperparameters (Full model)

| Symbol / name | Value |
|---|---|
| Iterations \(T\) | 30{,}000 (each stage) |
| Flat experts \(E\) / root leaves \(B\) | 8 |
| Top-\(k\) | 2 |
| Curriculum \(T_{\mathrm{w}}\) | 3{,}000 |
| \(\lambda_{\mathrm{op}},\lambda_{\mathrm{load}},\lambda_{\mathrm{ent}}\) | 0.01, 0.005, 0.002 |
| Grow checks | 6k, 10k, 14k, 18k |
| \(G_{\max}\), \(D_{\max}\), \(L_{\max}\) | 4, 3, 14 |
| Grow factor \(G\) | 2 |
| Max heads / leaves per event | 2 / 1 |
| Post-grow dense \(T_{\mathrm{dense}}\) | 1{,}500 |

## Appendix B — Suggested paper title

**Growing Tree Mixture-of-Experts for Scaffold-Gaussian Splatting**

---

*Numbers sourced from `ablations_all_scenes.csv`, `ablate_tree_grows_all_scenes.csv`, and `compute_cost.md` (regenerated Sep 2026).*
