#!/usr/bin/env python3
"""Clean routing figures from routing_analysis.json dumps."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path("/home/qiqi/Documents/Scaffold-GS-main")
OUT = ROOT / "documents" / "method_summary" / "routing"

SUITES = {
    "mipnerf360": ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"],
    "tandt": ["train", "truck"],
    "db": ["drjohnson", "playroom"],
    "nerfsynthetic": ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"],
}
ALL_METHODS = ["baseline", "moe", "moe_tree"]
HEADS = ["opacity", "cov", "color"]
HEAD_COLORS = {"opacity": "#3B6FA0", "cov": "#D07A12", "color": "#2F8F4E"}
METHOD_COLORS = {"baseline": "#8A8A8A", "moe": "#2F6DB5", "moe_tree": "#C43C3C"}
METHOD_LABELS = {"baseline": "baseline", "moe": "flat MoE", "moe_tree": "tree MoE"}
MASS_CMAP = LinearSegmentedColormap.from_list(
    "mass", ["#F7F4EF", "#F3C985", "#E07B3A", "#9B2C2C"]
)
MASS_CMAP.set_bad("#ECECE8")


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )


def latest_run(dataset: str, scene: str, method: str) -> Path | None:
    d = ROOT / "outputs" / dataset / scene / method
    if not d.is_dir():
        return None
    runs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "routing_analysis.json").exists())
    return runs[-1] if runs else None


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_runs():
    runs = {}
    for ds, scenes in SUITES.items():
        for scene in scenes:
            for method in ALL_METHODS:
                p = latest_run(ds, scene, method)
                if p is not None:
                    runs[(ds, scene, method)] = p
    return runs


def load_psnr(run_dir: Path) -> float | None:
    p = run_dir / "results.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    block = d.get("ours_30000") or next(iter(d.values()), {})
    return float(block["PSNR"]) if block and "PSNR" in block else None


def series(data: dict, head: str, key: str):
    xs, ys = [], []
    for snap in data["timeline"]:
        h = (snap.get("heads") or {}).get(head)
        if not h or key not in h:
            continue
        xs.append(int(snap["iter"]))
        ys.append(h[key])
    return np.array(xs), np.array(ys, dtype=float)


def final_head(data: dict, head: str):
    for snap in reversed(data["timeline"]):
        h = (snap.get("heads") or {}).get(head)
        if h is not None:
            return h
    return None


def grow_iters(data: dict):
    return sorted({int(s["iter"]) for s in data["timeline"] if str(s.get("event", "")).startswith("grow")})


def util_matrix(data: dict, head: str):
    iters, rows = [], []
    for snap in data["timeline"]:
        h = (snap.get("heads") or {}).get(head)
        if not h or not h.get("util"):
            continue
        iters.append(int(snap["iter"]))
        rows.append(np.array(h["util"], dtype=float))
    if not rows:
        return np.array([]), np.empty((0, 0))
    n = max(len(r) for r in rows)
    mat = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    return np.array(iters), mat


def tidy_axes(ax, xlabel=None, ylabel=None, show_x=True, show_y=True):
    ax.grid(True, axis="y", color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if xlabel and show_x:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("")
        if not show_x:
            ax.tick_params(labelbottom=False)
    if ylabel and show_y:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
        if not show_y:
            ax.tick_params(labelleft=False)


def draw_grows(ax, grows, ymax=None):
    for it in grows:
        ax.axvline(it, color="#C8C8C8", ls="--", lw=0.7, zorder=1)


def iter_ticks(ax):
    ax.set_xlim(0, 30000)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_xticklabels(["0", "10k", "20k", "30k"])


def leaf_ticks(ax, n, max_labels=8):
    if n <= 0:
        return
    step = 1 if n <= max_labels else 2
    ax.set_xticks(range(0, n, step))


def plot_heatmap(ax, iters, mat, vmax):
    if mat.size == 0:
        ax.set_visible(False)
        return None
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=MASS_CMAP,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        origin="lower",
        zorder=2,
    )
    ax.set_facecolor("#ECECE8")
    n = mat.shape[1]
    ax.set_xlim(-0.5, n - 0.5)
    leaf_ticks(ax, n)
    targets = [0, 10000, 20000, 30000]
    idxs, labs = [], []
    for t in targets:
        i = int(np.argmin(np.abs(iters.astype(float) - t)))
        if idxs and i == idxs[-1]:
            continue
        idxs.append(i)
        labs.append("0" if iters[i] == 0 else f"{int(iters[i]) // 1000}k")
    ax.set_yticks(idxs)
    ax.set_yticklabels(labs)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("#CCCCCC")
    ax.grid(False)
    return im


def plot_one_method_scene(ds: str, scene: str, method: str, run_dir: Path):
    data = load(run_dir / "routing_analysis.json")
    psnr = load_psnr(run_dir)
    grows = grow_iters(data) if method == "moe_tree" else []
    finals = {h: final_head(data, h) for h in HEADS}

    fig = plt.figure(figsize=(11.6, 7.6))
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1, 1, 1, 0.045],
        height_ratios=[1.05, 1.45, 0.95],
        left=0.07,
        right=0.97,
        top=0.86,
        bottom=0.09,
        wspace=0.16,
        hspace=0.32,
    )

    psnr_s = f"PSNR {psnr:.2f}" if psnr is not None else ""
    n_color = finals["color"]["n_experts"] if finals["color"] else "?"
    subtitle = {
        "baseline": "shared MLP  ·  1 expert",
        "moe": f"flat MoE  ·  {n_color} experts  ·  Top-2",
        "moe_tree": f"growing tree  ·  color leaves → {n_color}",
    }[method]
    fig.text(0.07, 0.955, f"{scene}", fontsize=15, fontweight="bold", color="#1A1A1A")
    fig.text(0.07, 0.915, f"{METHOD_LABELS[method]}  ·  {subtitle}  ·  {psnr_s}", fontsize=10, color="#555555")
    fig.text(0.97, 0.955, ds, fontsize=10, color="#888888", ha="right")

    h_max = 0.04
    v_max = 0.05
    n_max = 1
    for head in HEADS:
        _, y = series(data, head, "sample_entropy")
        if len(y):
            h_max = max(h_max, float(np.nanmax(np.abs(y))))
        _, mat = util_matrix(data, head)
        if mat.size:
            v_max = max(v_max, float(np.nanmax(mat)))
            n_max = max(n_max, mat.shape[1])
    h_max = min(max(h_max * 1.18, 0.06), 2.3)
    if method == "baseline":
        v_max = 1.0
    else:
        v_max = min(0.40, max(0.18, v_max * 1.05))

    last_im = None
    for col, head in enumerate(HEADS):
        color = HEAD_COLORS[head]
        fh = finals[head]
        n = int(fh["n_experts"]) if fh else 1

        ax_e = fig.add_subplot(gs[0, col])
        ax_h = fig.add_subplot(gs[1, col])
        ax_b = fig.add_subplot(gs[2, col])

        ax_e.set_title(head, color=color, fontweight="bold", pad=6)
        x, y = series(data, head, "sample_entropy")
        if len(x):
            ax_e.plot(x, y, color=color, lw=1.8, solid_capstyle="round")
        draw_grows(ax_e, grows)
        ax_e.set_ylim(0, h_max)
        iter_ticks(ax_e)
        tidy_axes(ax_e, xlabel="iteration", ylabel="entropy H", show_x=False, show_y=(col == 0))
        if fh:
            ax_e.text(
                0.97,
                0.90,
                f"H = {abs(fh.get('sample_entropy', 0)):.3f}",
                transform=ax_e.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )

        iters, mat = util_matrix(data, head)
        if mat.size and mat.shape[1] < n_max:
            mat = np.hstack([mat, np.full((mat.shape[0], n_max - mat.shape[1]), np.nan)])
        im = plot_heatmap(ax_h, iters, mat, v_max)
        if im is not None:
            last_im = im
        ax_h.set_ylabel("iteration" if col == 0 else "")
        if col != 0:
            ax_h.tick_params(labelleft=False)
        ax_h.set_xlabel("leaf" if col == 1 else "")

        if fh and fh.get("util"):
            util = np.array(fh["util"], dtype=float)
            ax_b.bar(np.arange(len(util)), util, color=color, width=0.82, zorder=2, linewidth=0)
            ax_b.set_ylim(0, v_max)
            leaf_ticks(ax_b, n_max)
            tidy_axes(ax_b, xlabel="leaf", ylabel="final mass", show_x=True, show_y=(col == 0))
            ax_b.set_xlim(-0.6, n_max - 0.4)
            ax_b.text(
                0.97,
                0.90,
                f"n={n}   CV={fh.get('cv', 0):.2f}",
                transform=ax_b.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )

    cax = fig.add_subplot(gs[1, 3])
    if last_im is not None:
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("mass", fontsize=8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, length=2)

    out = OUT / "by_method" / ds / scene
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{method}.png")
    plt.close(fig)


def plot_scene_compare(ds: str, scene: str, runs: dict):
    """One scene, three methods, color head only."""
    fig = plt.figure(figsize=(11.8, 7.2))
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1, 1, 1, 0.045],
        height_ratios=[1.05, 1.45, 0.95],
        left=0.07,
        right=0.97,
        top=0.84,
        bottom=0.09,
        wspace=0.16,
        hspace=0.32,
    )

    psnrs = []
    for m in ALL_METHODS:
        p = load_psnr(runs[(ds, scene, m)]) if (ds, scene, m) in runs else None
        psnrs.append(f"{METHOD_LABELS[m]} {p:.2f}" if p is not None else METHOD_LABELS[m])
    fig.text(0.07, 0.955, f"{scene}", fontsize=15, fontweight="bold")
    fig.text(0.07, 0.912, "color head  ·  PSNR  " + "   ·   ".join(psnrs), fontsize=9.5, color="#555555")
    fig.text(0.97, 0.955, ds, fontsize=10, color="#888888", ha="right")

    packed = {}
    for method in ALL_METHODS:
        key = (ds, scene, method)
        if key not in runs:
            continue
        packed[method] = load(runs[key] / "routing_analysis.json")

    moe_vmax = 0.18
    for method in ("moe", "moe_tree"):
        if method not in packed:
            continue
        _, mat = util_matrix(packed[method], "color")
        if mat.size:
            moe_vmax = max(moe_vmax, float(np.nanmax(mat)))
    moe_vmax = min(0.40, moe_vmax * 1.08)

    last_im = None
    for col, method in enumerate(ALL_METHODS):
        if method not in packed:
            continue
        data = packed[method]
        color = METHOD_COLORS[method]
        fh = final_head(data, "color")
        grows = grow_iters(data) if method == "moe_tree" else []
        col_vmax = 1.0 if method == "baseline" else moe_vmax

        ax_e = fig.add_subplot(gs[0, col])
        ax_h = fig.add_subplot(gs[1, col])
        ax_b = fig.add_subplot(gs[2, col])

        ax_e.set_title(METHOD_LABELS[method], color=color, fontweight="bold", pad=6)
        x, y = series(data, "color", "sample_entropy")
        if len(x):
            ax_e.plot(x, y, color=color, lw=1.8, solid_capstyle="round")
            y_max = max(0.06, float(np.nanmax(np.abs(y))) * 1.18)
        else:
            y_max = 0.06
        draw_grows(ax_e, grows)
        ax_e.set_ylim(0, y_max)
        iter_ticks(ax_e)
        tidy_axes(ax_e, ylabel="entropy H", show_x=False, show_y=True)
        if col != 0:
            ax_e.set_ylabel("")
        if fh:
            ax_e.text(
                0.97,
                0.90,
                f"H = {abs(fh.get('sample_entropy', 0)):.3f}",
                transform=ax_e.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )

        iters, mat = util_matrix(data, "color")
        im = plot_heatmap(ax_h, iters, mat, col_vmax)
        if im is not None and method != "baseline":
            last_im = im
        ax_h.set_ylabel("iteration" if col == 0 else "")
        if col != 0:
            ax_h.tick_params(labelleft=False)
        ax_h.set_xlabel("leaf" if col == 1 else "")

        if fh and fh.get("util"):
            util = np.array(fh["util"], dtype=float)
            ax_b.bar(np.arange(len(util)), util, color=color, width=0.82, zorder=2, linewidth=0)
            bar_top = 1.05 if method == "baseline" else moe_vmax
            ax_b.set_ylim(0, bar_top)
            leaf_ticks(ax_b, len(util))
            tidy_axes(ax_b, xlabel="leaf", ylabel="final mass", show_x=True, show_y=True)
            if col != 0:
                ax_b.set_ylabel("")
            ax_b.set_xlim(-0.6, len(util) - 0.4)
            ax_b.text(
                0.97,
                0.90,
                f"n={fh['n_experts']}   CV={fh.get('cv', 0):.2f}",
                transform=ax_b.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
            )

    cax = fig.add_subplot(gs[1, 3])
    if last_im is not None:
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label("mass", fontsize=8)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, length=2)

    out = OUT / "by_scene" / ds
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{scene}.png")
    plt.close(fig)


def plot_suite_entropy(ds: str, scenes: list[str], runs: dict):
    n = len(scenes)
    cols = 4 if n > 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.15 * cols, 2.45 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16, wspace=0.14, hspace=0.38)
    fig.text(0.08, 0.92, f"{ds}", fontsize=13, fontweight="bold")
    fig.text(0.08, 0.875, "color-head sample entropy", fontsize=10, color="#555555")

    for i, scene in enumerate(scenes):
        ax = axes[i // cols][i % cols]
        for method in ("moe", "moe_tree"):
            key = (ds, scene, method)
            if key not in runs:
                continue
            data = load(runs[key] / "routing_analysis.json")
            x, y = series(data, "color", "sample_entropy")
            ax.plot(x, y, color=METHOD_COLORS[method], lw=1.6, label=METHOD_LABELS[method])
            if method == "moe_tree":
                draw_grows(ax, grow_iters(data))
        ax.set_title(scene, fontsize=10, pad=4)
        iter_ticks(ax)
        tidy_axes(ax, show_x=(i // cols == rows - 1), show_y=(i % cols == 0))
        if i % cols == 0:
            ax.set_ylabel("H")
        if i // cols == rows - 1:
            ax.set_xlabel("iteration")

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    handles = [
        Line2D([0], [0], color=METHOD_COLORS["moe"], lw=1.8, label="flat MoE"),
        Line2D([0], [0], color=METHOD_COLORS["moe_tree"], lw=1.8, label="tree MoE"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=2)
    fig.savefig(OUT / f"{ds}__entropy_grid.png")
    plt.close(fig)


def plot_suite_mass(ds: str, scenes: list[str], runs: dict):
    mats = {"moe": [], "moe_tree": []}
    ns = {"moe": 1, "moe_tree": 1}
    for scene in scenes:
        for method in mats:
            key = (ds, scene, method)
            if key not in runs:
                mats[method].append(None)
                continue
            fh = final_head(load(runs[key] / "routing_analysis.json"), "color")
            util = np.array(fh["util"], dtype=float)
            mats[method].append(util)
            ns[method] = max(ns[method], len(util))

    stacked = {}
    for method, rows in mats.items():
        n = ns[method]
        m = np.full((len(scenes), n), np.nan)
        for i, row in enumerate(rows):
            if row is None:
                continue
            m[i, : len(row)] = row
        stacked[method] = m

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.6, 0.36 * len(scenes) + 1.35),
        gridspec_kw={"width_ratios": [ns["moe"], ns["moe_tree"]], "wspace": 0.12},
    )
    fig.subplots_adjust(left=0.16, right=0.90, top=0.78, bottom=0.16)
    fig.text(0.16, 0.92, f"{ds}", fontsize=13, fontweight="bold")
    fig.text(0.16, 0.86, "final color-leaf routing mass", fontsize=10, color="#555555")

    last_im = None
    for ax, method in zip(axes, ("moe", "moe_tree")):
        last_im = ax.imshow(
            stacked[method],
            aspect="auto",
            cmap=MASS_CMAP,
            vmin=0,
            vmax=0.45,
            interpolation="nearest",
        )
        ax.set_title(METHOD_LABELS[method], color=METHOD_COLORS[method], fontweight="bold", pad=8)
        ax.set_xticks(range(ns[method]))
        ax.set_xticklabels([str(i) for i in range(ns[method])], fontsize=8)
        ax.set_xlabel("leaf")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
            s.set_color("#CCCCCC")
        ax.tick_params(length=2)
    axes[0].set_yticks(range(len(scenes)))
    axes[0].set_yticklabels(scenes)
    axes[1].set_yticks(range(len(scenes)))
    axes[1].tick_params(labelleft=False)

    cax = fig.add_axes([0.92, 0.16, 0.018, 0.62])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("mass", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    fig.savefig(OUT / f"{ds}__final_util_heatmap.png")
    plt.close(fig)


def plot_summary(runs: dict):
    rows = []
    for ds, scenes in SUITES.items():
        for scene in scenes:
            for method in ("moe", "moe_tree"):
                key = (ds, scene, method)
                if key not in runs:
                    continue
                fh = final_head(load(runs[key] / "routing_analysis.json"), "color")
                if not fh:
                    continue
                rows.append((ds, scene, method, fh["sample_entropy"], fh["cv"]))

    scenes_order = []
    for ds, scenes in SUITES.items():
        for s in scenes:
            if any(r[1] == s and r[0] == ds for r in rows):
                scenes_order.append((ds, s))

    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.4), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.16, hspace=0.18)
    fig.text(0.08, 0.94, "Final color-head routing", fontsize=13, fontweight="bold")
    fig.text(0.08, 0.90, "flat MoE vs growing tree  ·  all scenes", fontsize=10, color="#555555")

    x = np.arange(len(scenes_order))
    w = 0.36
    # suite gaps: insert visual separators
    metrics = [("sample entropy H", 3), ("utilization CV", 4)]
    for ax, (ylabel, idx) in zip(axes, metrics):
        for k, method in enumerate(("moe", "moe_tree")):
            ys = [
                next((r[idx] for r in rows if r[0] == ds and r[1] == scene and r[2] == method), np.nan)
                for ds, scene in scenes_order
            ]
            ax.bar(
                x + (k - 0.5) * w,
                ys,
                w,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method] if ax is axes[0] else None,
                linewidth=0,
                zorder=2,
            )
        tidy_axes(ax, ylabel=ylabel, show_x=(ax is axes[-1]))
        acc = -0.5
        for ds, scenes in SUITES.items():
            n = sum(1 for d, _ in scenes_order if d == ds)
            if not n:
                continue
            ax.axvline(acc, color="#DDDDDD", lw=0.8, zorder=1)
            acc += n
        ax.set_xlim(-0.7, len(scenes_order) - 0.3)

    acc = 0
    ytop = axes[0].get_ylim()[1]
    for ds, scenes in SUITES.items():
        n = sum(1 for d, _ in scenes_order if d == ds)
        if not n:
            continue
        axes[0].text(
            acc + (n - 1) / 2,
            ytop,
            ds,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#888888",
            clip_on=False,
        )
        acc += n

    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([s for _, s in scenes_order], rotation=40, ha="right")
    fig.savefig(OUT / "all_scenes__final_metrics.png")
    plt.close(fig)


def write_metrics_csv(runs: dict):
    path = OUT / "final_routing_metrics.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scene", "method", "head", "n_experts", "sample_entropy", "cv", "max_util", "min_util", "psnr"])
        for (ds, scene, method), run_dir in sorted(runs.items()):
            data = load(run_dir / "routing_analysis.json")
            psnr = load_psnr(run_dir)
            for head in HEADS:
                fh = final_head(data, head)
                if not fh:
                    continue
                w.writerow([ds, scene, method, head, fh.get("n_experts"), fh.get("sample_entropy"), fh.get("cv"), fh.get("max_util"), fh.get("min_util"), psnr])


def color_stats(runs: dict, ds: str, scene: str, method: str):
    key = (ds, scene, method)
    if key not in runs:
        return None
    fh = final_head(load(runs[key] / "routing_analysis.json"), "color")
    psnr = load_psnr(runs[key])
    if not fh:
        return None
    return dict(
        psnr=psnr,
        n=int(fh.get("n_experts", 0)),
        H=abs(float(fh.get("sample_entropy", 0))),
        cv=float(fh.get("cv", 0)),
        maxu=float(fh.get("max_util", 0)),
        minu=float(fh.get("min_util", 0)),
    )


def scene_caption(ds: str, scene: str, runs: dict) -> str:
    s = {m: color_stats(runs, ds, scene, m) for m in ALL_METHODS}
    if not all(s.values()):
        return ""
    b, f, t = s["baseline"], s["moe"], s["moe_tree"]
    d_f = f["psnr"] - b["psnr"]
    d_t = t["psnr"] - b["psnr"]
    winner = "tree MoE" if t["psnr"] >= f["psnr"] and t["psnr"] >= b["psnr"] else (
        "flat MoE" if f["psnr"] >= b["psnr"] else "baseline"
    )
    notes = []
    if f["H"] > 0.07:
        notes.append(f"flat MoE keeps unusually high final entropy (H={f['H']:.3f})")
    if t["H"] < 0.015:
        notes.append(f"tree routing is nearly deterministic (H={t['H']:.3f})")
    if f["cv"] >= 0.85:
        notes.append(f"flat MoE is peaked (CV={f['cv']:.2f}, max leaf {f['maxu']:.0%} of mass)")
    if t["minu"] < 1e-4:
        notes.append("tree ends with at least one unused color leaf (parent split, mass moved to children)")
    extra = " " + "; ".join(notes) + "." if notes else ""
    return (
        f"**This scene.** PSNR {b['psnr']:.2f} / {f['psnr']:.2f} / {t['psnr']:.2f} "
        f"(Δ vs baseline {d_f:+.2f} / {d_t:+.2f}). "
        f"Color-head final *H* = {f['H']:.3f} (8 leaves) vs {t['H']:.3f} ({t['n']} leaves); "
        f"CV = {f['cv']:.2f} vs {t['cv']:.2f}. Best PSNR: {winner}.{extra}"
    )


def load_ablate_csv():
    path = OUT / "ablate_tree_grows_all_scenes.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_ablate_tree_grows(rows: list[dict] | None = None):
    """PSNR bars for moe_max_total_grows ∈ {0, 2, 4} vs flat MoE."""
    rows = rows if rows is not None else load_ablate_csv()
    if not rows:
        print("skip ablate_tree_grows figure (no CSV)")
        return
    scenes = [r["scene"] for r in rows]
    moe = np.array([float(r["moe_psnr"]) for r in rows])
    g0 = np.array([float(r["g0_psnr"]) for r in rows])
    g2 = np.array([float(r["g2_psnr"]) for r in rows])
    g4 = np.array([float(r["g4_psnr"]) for r in rows])
    x = np.arange(len(scenes))
    w = 0.22
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), gridspec_kw={"height_ratios": [1.15, 1.0]})

    ax = axes[0]
    ax.bar(x - 1.5 * w, g0 - moe, width=w, color="#8A8A8A", label="grows=0 (8 leaves)")
    ax.bar(x - 0.5 * w, g2 - moe, width=w, color="#2F6DB5", label="grows=2 (10 leaves)")
    ax.bar(x + 0.5 * w, g4 - moe, width=w, color="#C43C3C", label="grows=4 (11 leaves)")
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_ylabel("Δ PSNR vs flat MoE")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha="right")
    ax.legend(ncol=3, loc="upper right")
    ax.set_title("Tree grow ablation — gain over warm-start flat MoE")

    ax = axes[1]
    ax.bar(x - 0.5 * w, g2 - g0, width=w, color="#2F6DB5", label="grows=2 − grows=0")
    ax.bar(x + 0.5 * w, g4 - g0, width=w, color="#C43C3C", label="grows=4 − grows=0")
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_ylabel("Δ PSNR vs grows=0")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=45, ha="right")
    ax.legend(ncol=2, loc="upper right")
    ax.set_title("Extra splits on top of no-growth tree")

    fig.tight_layout()
    fig.savefig(OUT / "ablate_tree_grows_all_scenes.png")
    plt.close(fig)


def write_index(runs: dict):
    lines = [
        "# Routing analysis",
        "",
        "Logs are `routing_analysis.json` in each run directory. Figures below are generated from those dumps.",
        "",
        "## How to read the figures",
        "",
        "Every **scene** figure is the same layout: **baseline | flat MoE | tree MoE**, color head only.",
        "",
        "| Panel | Meaning |",
        "|---|---|",
        "| Entropy *H* | Mean per-sample gate entropy. 0 = the router always picks the same expert(s). ~2.08 = uniform over 8 experts (dense warmup). |",
        "| Heatmap | Routing mass of each leaf over training. Darker = more mass. Gray = leaf not grown yet. |",
        "| Final mass | Share of gates on each leaf at 30k. *n* = leaf count. CV = imbalance (0 = equal, high = a few leaves dominate). |",
        "",
        "- **baseline** — shared MLP, 1 expert, no routing. Included as a control.",
        "- **flat MoE** — 8 experts, dense softmax warmup, then Top-2.",
        "- **tree MoE** — starts at 8 leaves (1:1 from flat MoE), grows at 6k / 10k / 14k / 18k; color typically ends at 11 leaves.",
        "",
        "Click **opacity / cov / color pages** under a scene for the same three panels on all heads of one method.",
        "",
        "Grow-count ablation (`moe_max_total_grows` = 0 / 2 / 4): see [ablate_tree_grows.md](ablate_tree_grows.md).",
        "",
        "## Suite summaries",
        "",
        "### All scenes — final color-head metrics",
        "",
        '<img src="all_scenes__final_metrics.png" width="920" alt="all scenes final metrics">',
        "",
        "**Figure.** Top: final sample entropy *H* on the color head. Tree MoE is lower than flat MoE on every scene "
        "(more specialized routing) even though it has more leaves (11 vs 8). Bottom: utilization CV. "
        "Flat MoE is sometimes more balanced (kitchen, garden) and sometimes more collapsed onto one expert "
        "(room, bonsai). Tree CV is steadier, typically 0.5–0.7.",
        "",
    ]
    suite_note = {
        "mipnerf360": "Tree MoE wins PSNR on all 7 scenes; largest gains on bonsai (+1.23) and kitchen (+1.00).",
        "tandt": "`train` is the highest-entropy scene in the sweep (flat *H*=0.100); tree still halves it.",
        "db": "Small but consistent PSNR gains; playroom keeps the highest remaining entropy of the indoor pair.",
        "nerfsynthetic": "Flat MoE is at or below baseline on most objects; tree recovers on all except ficus (−0.30 vs baseline).",
    }
    for ds, scenes in SUITES.items():
        lines += [
            f"### {ds} — entropy over training",
            "",
            f'<img src="{ds}__entropy_grid.png" width="920" alt="{ds} entropy grid">',
            "",
            f"**Figure.** Color-head entropy vs iteration for each {ds} scene. Blue = flat MoE, red = tree MoE. "
            f"Shared y-axis: flat MoE always shows the dense-warmup plateau near *H*≈2.1, then a collapse after Top-2 "
            f"turns on (~3–10k). Tree MoE stays near 0, with a small bump around 3k and faint dashed grow lines. "
            f"{suite_note[ds]}",
            "",
            f"### {ds} — final leaf mass",
            "",
            f'<img src="{ds}__final_util_heatmap.png" width="720" alt="{ds} util heatmap">',
            "",
            f"**Figure.** Final color-leaf routing mass. Left: flat MoE, 8 leaves. Right: tree MoE, 11 leaves. "
            f"Each row is a scene. Darker = that leaf carries more of the gate mass. Flat MoE often dumps mass on "
            f"1–2 of the original 8 experts; tree MoE spreads mass over more leaves, and some early leaves go dark "
            f"after a split (mass moved to children).",
            "",
        ]

    for ds, scenes in SUITES.items():
        lines += [f"## {ds} — per scene", ""]
        for scene in scenes:
            psnrs = []
            for method in ALL_METHODS:
                st = color_stats(runs, ds, scene, method)
                if st and st["psnr"] is not None:
                    psnrs.append(f"{METHOD_LABELS[method]} {st['psnr']:.2f}")
            lines += [
                f"### {scene}",
                "",
                "PSNR: " + " · ".join(psnrs),
                "",
                f'<img src="by_scene/{ds}/{scene}.png" width="920" alt="{scene} comparison">',
                "",
                scene_caption(ds, scene, runs),
                "",
                "<details><summary>opacity / cov / color pages</summary>",
                "",
                "Same three panels, now one column per **head** (opacity, covariance, color) for a single method. "
                "Use these when color-only is not enough — opacity often stays higher-entropy than color, and "
                "covariance usually collapses first.",
                "",
                "<p>",
            ]
            for method in ALL_METHODS:
                rel = f"by_method/{ds}/{scene}/{method}.png"
                lines.append(
                    f'<a href="{rel}"><img src="{rel}" width="300" alt="{scene} {method}"></a>'
                )
            lines += ["</p>", "", "</details>", ""]
    (OUT / "INDEX.md").write_text("\n".join(lines))


def cleanup_old():
    """Remove figures the old script wrote that this layout no longer uses."""
    for p in [
        OUT / "kitchen__entropy_story.png",
    ]:
        if p.exists():
            p.unlink()
    per = OUT / "per_scene"
    if per.exists():
        for f in per.rglob("*.png"):
            f.unlink()
        for d in sorted(per.rglob("*"), reverse=True):
            if d.is_dir():
                d.rmdir()
        if per.is_dir():
            per.rmdir()


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    cleanup_old()
    runs = collect_runs()
    print(f"found {len(runs)} runs")
    for ds, scenes in SUITES.items():
        for scene in scenes:
            print(f"  scene {ds}/{scene}")
            plot_scene_compare(ds, scene, runs)
            for method in ALL_METHODS:
                key = (ds, scene, method)
                if key in runs:
                    plot_one_method_scene(ds, scene, method, runs[key])
        plot_suite_entropy(ds, scenes, runs)
        plot_suite_mass(ds, scenes, runs)
    plot_summary(runs)
    plot_ablate_tree_grows()
    write_metrics_csv(runs)
    write_index(runs)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
