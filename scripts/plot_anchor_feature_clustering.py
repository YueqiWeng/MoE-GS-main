#!/usr/bin/env python3
"""Anchor-feature clustering experiment (Scaffold-GS Fig. 6 protocol).

Lu et al., CVPR 2024, Fig. 6 / Sec. 4.2 Feature Analysis:
  cluster stored anchor features f_v in R^{32} with K-means (K=3) on
  Deep Blending playroom, then color each anchor in 3D by cluster id.
  They use the *stored* local context feature f_v, not the view-blended
  hat{f}_v from the feature bank.

This script repeats that analysis on finished baseline / moe / moe_tree
runs (same 30k ply). Optional: compare K-means labels to the color-head
router assignment (argmax leaf on z=f_v).
"""
from __future__ import annotations

import csv
import os
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/qiqi/Documents/Scaffold-GS-main")
OUT = ROOT / "documents" / "method_summary" / "routing"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ["baseline", "moe", "moe_tree"]
PAPER_METHODS = ["3dgs", "baseline", "moe", "moe_tree"]
METHOD_LABELS = {
    "3dgs": "3D-GS",
    "baseline": "Scaffold-GS",
    "moe": "flat MoE",
    "moe_tree": "tree MoE",
}
# Wong colorblind-safe (3 clusters, same as Scaffold-GS K=3).
CLUSTER_RGB = np.array(
    [
        [213, 94, 0],
        [0, 158, 115],
        [0, 114, 178],
        [204, 121, 167],
        [230, 159, 0],
        [86, 180, 233],
        [240, 228, 66],
        [0, 0, 0],
        [128, 128, 128],
        [170, 68, 153],
        [153, 153, 51],
        [102, 51, 0],
    ],
    dtype=np.float64,
) / 255.0

# 2x4 paper figure: original RGB | Scaffold-GS | flat MoE | Tree MoE.
SCENES = [
    {
        "dataset": "db",
        "scene": "playroom",
        "photo": "00024.png",
        "elev": 18,
        "azim": -70,
        "pca": False,
        "ps": 0.40,
        "inlier_lo": 4.0,
        "inlier_hi": 96.0,
    },
    {
        "dataset": "mipnerf360",
        "scene": "kitchen",
        "photo": "00008.png",
        "elev": 88,
        "azim": -90,
        "pca": True,
        "ps": 0.85,
        "inlier_lo": 8.0,
        "inlier_hi": 92.0,
    },
]


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.facecolor": "white",
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


def iteration_ply(run: Path, method: str) -> Path | None:
    if method == "3dgs":
        p = run / "gs_run" / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        if p.exists():
            return p
    p = run / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    return p if p.exists() else None


def latest_run(dataset: str, scene: str, method: str) -> Path:
    d = ROOT / "outputs" / dataset / scene / method
    if not d.is_dir():
        raise FileNotFoundError(d)
    runs = sorted(
        [
            p
            for p in d.iterdir()
            if p.is_dir() and (p / "results.json").exists() and iteration_ply(p, method) is not None
        ]
    )
    if not runs:
        raise FileNotFoundError(f"no 30k ply for {dataset}/{scene}/{method}")
    return runs[-1]


def read_3dgs_ply(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """3D-GS has no f_v. Cluster SH coefficients of each Gaussian (appearance)."""
    ply = PlyData.read(str(ply_path))
    el = ply.elements[0]
    xyz = np.stack(
        [np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])],
        axis=1,
    ).astype(np.float64)
    dc = sorted(
        [p.name for p in el.properties if p.name.startswith("f_dc_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    rest = sorted(
        [p.name for p in el.properties if p.name.startswith("f_rest_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    names = dc + rest
    if not names:
        raise RuntimeError(f"{ply_path}: no SH features")
    feat = np.stack([np.asarray(el[n]) for n in names], axis=1).astype(np.float64)
    return xyz, feat


def read_anchor_ply(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz [N,3], f_v [N,32]) from a Scaffold-GS sparse-Gaussian ply."""
    ply = PlyData.read(str(ply_path))
    el = ply.elements[0]
    xyz = np.stack(
        [np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])],
        axis=1,
    ).astype(np.float64)
    names = sorted(
        [p.name for p in el.properties if p.name.startswith("f_anchor_feat")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if len(names) != 32:
        raise RuntimeError(f"{ply_path}: expected 32-D f_v, got {len(names)}")
    feat = np.stack([np.asarray(el[n]) for n in names], axis=1).astype(np.float64)
    return xyz, feat


def spatial_inliers(xyz: np.ndarray, lo: float = 4.0, hi: float = 96.0) -> np.ndarray:
    """Drop far floaters so the 3D plot matches the room, not the bbox."""
    p_lo = np.percentile(xyz, lo, axis=0)
    p_hi = np.percentile(xyz, hi, axis=0)
    return np.all((xyz >= p_lo) & (xyz <= p_hi), axis=1)


def pca_orient(xyz: np.ndarray) -> np.ndarray:
    """Put the two largest-variance axes on x/y so the room fills the frame."""
    mu = xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(xyz - mu, full_matrices=False)
    return (xyz - mu) @ vt.T


def kmeans_labels(feat: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Lloyd K-means on raw f_v (Scaffold-GS does not whitening-normalize)."""
    rng = np.random.default_rng(seed)
    # k-means++ init via a seeded numpy RNG, then scipy Lloyd.
    n = feat.shape[0]
    centers = np.empty((k, feat.shape[1]), dtype=np.float64)
    idx0 = int(rng.integers(0, n))
    centers[0] = feat[idx0]
    closest = np.full(n, np.inf)
    for j in range(1, k):
        d2 = ((feat - centers[j - 1]) ** 2).sum(axis=1)
        closest = np.minimum(closest, d2)
        p = closest / closest.sum().clip(min=1e-12)
        centers[j] = feat[int(rng.choice(n, p=p))]
    _, labels = kmeans2(feat, centers, iter=40, minit="matrix", missing="warn")
    return labels.astype(np.int64)


def align_labels_3d(
    xyz: np.ndarray, labels: np.ndarray, xyz_ref: np.ndarray, labels_ref: np.ndarray, k: int
) -> np.ndarray:
    """Permute cluster ids so 3D centroids match the reference (baseline)."""

    def cents(x, lab):
        out = np.zeros((k, 3), dtype=np.float64)
        for i in range(k):
            m = lab == i
            out[i] = x[m].mean(axis=0) if m.any() else np.full(3, 1e6)
        return out

    cost = ((cents(xyz, labels)[:, None, :] - cents(xyz_ref, labels_ref)[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    mapping = {int(r[i]): int(c[i]) for i in range(len(r))}
    return np.array([mapping[int(v)] for v in labels], dtype=np.int64)


def subsample(n: int, cap: int, seed: int = 0) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    if n == 0:
        return 0.0

    def ent(lab):
        _, cnt = np.unique(lab, return_counts=True)
        p = cnt / n
        return float(-(p * np.log(p + 1e-12)).sum())

    mi = 0.0
    for ua in np.unique(a):
        ma = a == ua
        pa = ma.mean()
        for ub in np.unique(b):
            pab = np.mean(ma & (b == ub))
            if pab <= 0:
                continue
            pb = np.mean(b == ub)
            mi += pab * np.log((pab + 1e-12) / (pa * pb + 1e-12))
    hy, hz = ent(a), ent(b)
    return float(mi / np.sqrt(hy * hz + 1e-12))


def load_cfg(run: Path) -> dict:
    text = (run / "cfg_args").read_text().strip()
    ns = eval(text, {"Namespace": Namespace})  # noqa: S307
    return dict(vars(ns))


def color_router_labels(run: Path, feat: np.ndarray) -> np.ndarray | None:
    """Argmax color-leaf on z=f_v (no view, no GNN). None if baseline / load fail."""
    ckpt_dir = run / "point_cloud" / "iteration_30000"
    if not (ckpt_dir / "checkpoints.pth").exists():
        return None
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from scene.gaussian_model import GaussianModel  # noqa: E402

    cfg = load_cfg(run)
    gaussians = GaussianModel(
        cfg.get("feat_dim", 32),
        cfg.get("n_offsets", 10),
        cfg.get("voxel_size", 0.001),
        cfg.get("update_depth", 3),
        cfg.get("update_init_factor", 16),
        cfg.get("update_hierachy_factor", 4),
        cfg.get("use_feat_bank", False),
        cfg.get("appearance_dim", 0),
        cfg.get("ratio", 1),
        cfg.get("add_opacity_dist", False),
        cfg.get("add_cov_dist", False),
        cfg.get("add_color_dist", False),
        cfg.get("num_experts", 1),
        cfg.get("moe_top_k", 2),
        cfg.get("use_gnn", False),
        cfg.get("gnn_hidden_dim", 32),
        cfg.get("gnn_layers", 2),
        cfg.get("gnn_k", 16),
        cfg.get("moe_arch", "flat"),
        cfg.get("moe_branch_factor", 4),
        cfg.get("moe_tree_depth", 2),
        cfg.get("moe_max_leaves", 10),
        cfg.get("moe_grow_factor", 0),
        cfg.get("moe_max_total_grows", 64),
        cfg.get("gnn_gamma_init", 0.5),
    )
    gaussians.load_ply_sparse_gaussian(str(ckpt_dir / "point_cloud.ply"))
    gaussians.load_mlp_checkpoints(str(ckpt_dir), mode="unite")
    moe = gaussians.mlp_color
    if getattr(moe, "num_experts", 1) <= 1:
        return None
    fv = torch.tensor(feat, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        if hasattr(moe, "root"):
            dummy = fv.new_zeros(fv.shape[0], moe.expert_in_dim)
            was = moe.training
            moe.train(True)
            moe.set_sparse(True)
            moe(fv, dummy, None)
            gates = moe._last_traffic_gates
            moe.train(was)
            labels = gates.argmax(dim=-1).cpu().numpy().astype(np.int64)
        else:
            labels = moe.router(fv).argmax(dim=-1).cpu().numpy().astype(np.int64)
    del gaussians, moe, fv
    torch.cuda.empty_cache()
    return labels


def set_equal_3d(ax, xyz: np.ndarray):
    c = xyz.mean(axis=0)
    r = 0.5 * (xyz.max(axis=0) - xyz.min(axis=0)).max()
    r = max(float(r), 1e-3)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-70)


def plot_scene_row(fig, row, n_rows, packed, spec, k: int):
    xyz_ref = packed["baseline"]["xyz"]
    lab_ref = packed["baseline"]["kmeans"]
    if spec.get("pca"):
        mu = xyz_ref.mean(axis=0)
        _, _, vt = np.linalg.svd(xyz_ref - mu, full_matrices=False)
        packed["_pca_mu"] = mu
        packed["_pca_vt"] = vt
    for col, method in enumerate(METHODS):
        ax = fig.add_subplot(n_rows, 3, row * 3 + col + 1, projection="3d")
        rec = packed[method]
        labels = align_labels_3d(rec["xyz"], rec["kmeans"], xyz_ref, lab_ref, k)
        idx = subsample(len(rec["xyz"]), 22000, seed=0)
        xyz = rec["xyz"][idx]
        if spec.get("pca"):
            xyz = (xyz - packed["_pca_mu"]) @ packed["_pca_vt"].T
        col_idx = np.clip(labels[idx], 0, len(CLUSTER_RGB) - 1)
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=CLUSTER_RGB[col_idx],
            s=spec.get("ps", 0.4),
            linewidths=0,
            alpha=0.9,
            rasterized=True,
        )
        set_equal_3d(ax, xyz)
        ax.view_init(elev=spec["elev"], azim=spec["azim"])
        if row == 0:
            ax.set_title(METHOD_LABELS[method], pad=2)
        if col == 0:
            ax.text2D(
                -0.08,
                0.5,
                spec["scene"],
                transform=ax.transAxes,
                va="center",
                ha="right",
                rotation=90,
                fontsize=10,
            )


def load_clustered(
    dataset: str, scene: str, method: str, k: int, lo: float = 4.0, hi: float = 96.0
) -> dict:
    run = latest_run(dataset, scene, method)
    ply = iteration_ply(run, method)
    assert ply is not None
    if method == "3dgs":
        xyz, feat = read_3dgs_ply(ply)
    else:
        xyz, feat = read_anchor_ply(ply)
    keep = spatial_inliers(xyz, lo=lo, hi=hi)
    xyz, feat = xyz[keep], feat[keep]
    # 3D-GS has ~2M Gaussians; cap k-means size.
    cap = 250000 if method == "3dgs" else len(xyz)
    if len(xyz) > cap:
        idx = subsample(len(xyz), cap, seed=0)
        xyz, feat = xyz[idx], feat[idx]
    labels = kmeans_labels(feat, k=k, seed=0)
    print(f"  [{method}] N={len(xyz)}  feat_dim={feat.shape[1]}")
    return {"run": run, "xyz": xyz, "kmeans": labels}


def original_photo(spec: dict) -> Path:
    run = latest_run(spec["dataset"], spec["scene"], "baseline")
    gt = run / "test" / "ours_30000" / "gt" / spec["photo"]
    if gt.exists():
        return gt
    hits = [p for p in (run / "test").rglob(spec["photo"]) if "gt" in p.parts]
    if not hits:
        raise FileNotFoundError(f"no GT {spec['photo']} under {run}/test")
    return hits[0]


def plot_matrix_2x3(k: int = 3) -> None:
    """Paper figure: 2 scenes x original | Scaffold-GS | flat MoE | Tree MoE."""
    packed_all = []
    for spec in SCENES:
        packed = {}
        print(f"== {spec['scene']}  K={k} on f_v ==")
        for method in METHODS:
            packed[method] = load_clustered(
                spec["dataset"],
                spec["scene"],
                method,
                k,
                lo=spec.get("inlier_lo", 4.0),
                hi=spec.get("inlier_hi", 96.0),
            )
        packed_all.append((spec, packed))

    n_rows = len(packed_all)
    fig = plt.figure(figsize=(13.8, 6.7))
    gs = GridSpec(
        n_rows,
        4,
        figure=fig,
        left=0.055,
        right=0.995,
        top=0.92,
        bottom=0.02,
        wspace=0.04,
        hspace=0.06,
        width_ratios=[1.15, 1, 1, 1],
    )
    col_titles = ["original"] + [METHOD_LABELS[m] for m in METHODS]
    for row, (spec, packed) in enumerate(packed_all):
        xyz_ref = packed["baseline"]["xyz"]
        lab_ref = packed["baseline"]["kmeans"]
        if spec.get("pca"):
            mu = xyz_ref.mean(axis=0)
            _, _, vt = np.linalg.svd(xyz_ref - mu, full_matrices=False)
            packed["_pca_mu"] = mu
            packed["_pca_vt"] = vt

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(np.array(Image.open(original_photo(spec)).convert("RGB")))
        ax0.set_axis_off()
        if row == 0:
            ax0.set_title(col_titles[0], pad=6)
        ax0.text(
            -0.04,
            0.5,
            spec["scene"],
            transform=ax0.transAxes,
            va="center",
            ha="right",
            rotation=90,
            fontsize=11,
        )

        for col, method in enumerate(METHODS):
            ax = fig.add_subplot(gs[row, col + 1], projection="3d")
            rec = packed[method]
            labels = align_labels_3d(rec["xyz"], rec["kmeans"], xyz_ref, lab_ref, k)
            idx = subsample(len(rec["xyz"]), 25000, seed=0)
            xyz = rec["xyz"][idx]
            if spec.get("pca"):
                xyz = (xyz - packed["_pca_mu"]) @ packed["_pca_vt"].T
            col_idx = np.clip(labels[idx], 0, len(CLUSTER_RGB) - 1)
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=CLUSTER_RGB[col_idx],
                s=spec.get("ps", 0.4),
                linewidths=0,
                alpha=0.9,
                rasterized=True,
            )
            set_equal_3d(ax, xyz)
            ax.view_init(elev=spec["elev"], azim=spec["azim"])
            if row == 0:
                ax.set_title(METHOD_LABELS[method], pad=6)
    png = OUT / "anchor_feature_clustering.png"
    pdf = OUT / "anchor_feature_clustering.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print("wrote", png)
    fig_dir = ROOT / "documents" / "method_summary" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    from shutil import copy2

    copy2(png, fig_dir / png.name)
    copy2(pdf, fig_dir / pdf.name)


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_matrix_2x3()


if __name__ == "__main__":
    main()
