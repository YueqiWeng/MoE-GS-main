#!/usr/bin/env python3
"""Color each anchor by the winning color-leaf (argmax router on z=f_v).

This is the Tree-MoE analysis, not Scaffold-GS Fig. 6 (K-means on f_v).
Same 3D cameras as the clustering figure. No new training: ply + color MLP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from shutil import copy2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image

ROOT = Path("/home/qiqi/Documents/Scaffold-GS-main")
sys.path.insert(0, str(ROOT / "scripts"))
import plot_anchor_feature_clustering as acf  # noqa: E402

OUT = ROOT / "documents" / "method_summary"
METHODS = ("moe", "moe_tree")
METHOD_LABELS = {"moe": "flat MoE", "moe_tree": "tree MoE"}


def leaf_legend(ax, labels: np.ndarray):
    """Color chips + leaf id + mass on this panel (raw expert index)."""
    from matplotlib.patches import Patch

    n = len(labels)
    u, cnt = np.unique(labels, return_counts=True)
    order = np.argsort(-cnt)
    handles = []
    for i in order:
        lid = int(u[i])
        frac = 100.0 * cnt[i] / n
        if frac < 0.5:
            continue
        rgb = acf.CLUSTER_RGB[lid % len(acf.CLUSTER_RGB)]
        handles.append(
            Patch(
                facecolor=rgb,
                edgecolor="none",
                label=f"L{lid}  {frac:.0f}%",
            )
        )
    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.02),
            fontsize=6.5,
            frameon=False,
            handlelength=0.8,
            handleheight=0.8,
            borderaxespad=0.1,
            labelspacing=0.15,
        )


def load_leaf_scene(spec: dict) -> dict:
    packed = {}
    lo, hi = spec.get("inlier_lo", 4.0), spec.get("inlier_hi", 96.0)
    for method in METHODS:
        run = acf.latest_run(spec["dataset"], spec["scene"], method)
        ply = acf.iteration_ply(run, method)
        xyz, feat = acf.read_anchor_ply(ply)
        keep = acf.spatial_inliers(xyz, lo=lo, hi=hi)
        xyz, feat = xyz[keep], feat[keep]
        labels = acf.color_router_labels(run, feat)
        if labels is None:
            raise RuntimeError(f"no color router on {run}")
        if len(labels) != len(xyz):
            raise RuntimeError(f"label/xyz mismatch {len(labels)} vs {len(xyz)}")
        u, cnt = np.unique(labels, return_counts=True)
        print(
            f"  [{method}] N={len(xyz)}  leaves={len(u)}  "
            f"top={u[np.argmax(cnt)]} ({100 * cnt.max() / len(labels):.1f}%)"
        )
        packed[method] = {"run": run, "xyz": xyz, "leaf": labels, "used": u.tolist()}
    return packed


def plot() -> None:
    acf.apply_style()
    packed_all = []
    for spec in acf.SCENES:
        print(f"== {spec['scene']}  winning color-leaf ==")
        packed_all.append((spec, load_leaf_scene(spec)))

    n_rows = len(packed_all)
    fig = plt.figure(figsize=(11.6, 7.2))
    gs = GridSpec(
        n_rows,
        3,
        figure=fig,
        left=0.055,
        right=0.995,
        top=0.90,
        bottom=0.04,
        wspace=0.06,
        hspace=0.10,
        width_ratios=[1.15, 1, 1],
    )
    summary = {}
    for row, (spec, packed) in enumerate(packed_all):
        xyz_ref = packed["moe"]["xyz"]
        lab_ref = packed["moe"]["leaf"]
        if spec.get("pca"):
            mu = xyz_ref.mean(axis=0)
            _, _, vt = np.linalg.svd(xyz_ref - mu, full_matrices=False)
            packed["_pca_mu"] = mu
            packed["_pca_vt"] = vt

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(np.array(Image.open(acf.original_photo(spec)).convert("RGB")))
        ax0.set_axis_off()
        if row == 0:
            ax0.set_title("original", pad=6)
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

        scene_sum = {}
        for col, method in enumerate(METHODS):
            ax = fig.add_subplot(gs[row, col + 1], projection="3d")
            rec = packed[method]
            labels = rec["leaf"]
            idx = acf.subsample(len(rec["xyz"]), 25000, seed=0)
            xyz = rec["xyz"][idx]
            if spec.get("pca"):
                xyz = (xyz - packed["_pca_mu"]) @ packed["_pca_vt"].T
            col_idx = np.clip(labels[idx], 0, len(acf.CLUSTER_RGB) - 1)
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                c=acf.CLUSTER_RGB[col_idx],
                s=spec.get("ps", 0.4),
                linewidths=0,
                alpha=0.9,
                rasterized=True,
            )
            acf.set_equal_3d(ax, xyz)
            ax.view_init(elev=spec["elev"], azim=spec["azim"])
            if row == 0:
                ax.set_title(METHOD_LABELS[method], pad=10)
            leaf_legend(ax, labels)
            u, cnt = np.unique(rec["leaf"], return_counts=True)
            scene_sum[method] = {
                "n_anchors": int(len(rec["xyz"])),
                "n_leaves_used": int(len(u)),
                "mass": {int(i): round(float(c) / len(rec["leaf"]), 4) for i, c in zip(u, cnt)},
            }
        summary[spec["scene"]] = scene_sum

    routing = OUT / "routing"
    figures = OUT / "figures"
    routing.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    png = routing / "winning_color_leaf.png"
    pdf = routing / "winning_color_leaf.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    copy2(png, figures / png.name)
    copy2(pdf, figures / pdf.name)
    meta = routing / "winning_color_leaf.json"
    meta.write_text(json.dumps(summary, indent=2))
    print("wrote", png)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    plot()
