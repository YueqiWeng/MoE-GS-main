#!/usr/bin/env python3
"""Scaffold-GS Fig. 9 analogue: activated neural Gaussians vs iterations for k."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/qiqi/Documents/Scaffold-GS-main")
OUT_DIR = ROOT / "documents" / "method_summary"
SCENE = "db/playroom"
KS = (5, 10, 20)
COLORS = {5: "#E0B54A", 10: "#C25555", 20: "#6AA84F"}
# Dense early, then roughly every 2–4k — same cadence as Scaffold-GS Fig. 9.
MARK_ITERS = np.array(
    [0, 500, 1000, 2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000]
)


def latest_run(k: int) -> Path:
    d = ROOT / "outputs" / SCENE / f"k_ablate_k{k}"
    runs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "activated_gaussians.json").exists())
    if not runs:
        raise FileNotFoundError(f"no activated_gaussians.json under {d}")
    return runs[-1]


def load_curve(run: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with open(run / "activated_gaussians.json") as f:
        payload = json.load(f)
    tl = payload["timeline"]
    it = np.array([r["iter"] for r in tl], dtype=float)
    y = np.array([r["activated_per_anchor"] for r in tl], dtype=float)
    order = np.argsort(it)
    it, y = it[order], y[order]
    psnr = float("nan")
    res_path = run / "results.json"
    if res_path.exists():
        with open(res_path) as f:
            res = json.load(f)
        psnr = float(res.get("ours_30000", {}).get("PSNR", float("nan")))
    return it, y, psnr


def rolling_mean(it: np.ndarray, y: np.ndarray, window: float = 400.0) -> np.ndarray:
    """Uniform window in iteration space (random-view samples are noisy)."""
    out = np.empty_like(y)
    for i, t in enumerate(it):
        m = (it >= t - window) & (it <= t + window)
        out[i] = y[m].mean()
    return out


def sample_marks(it: np.ndarray, y: np.ndarray, marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for t in marks:
        if t < it.min() - 50 or t > it.max() + 50:
            continue
        j = int(np.argmin(np.abs(it - t)))
        xs.append(it[j])
        ys.append(y[j])
    return np.array(xs), np.array(ys)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    curves = {}
    for k in KS:
        run = latest_run(k)
        it, y_raw, psnr = load_curve(run)
        y = rolling_mean(it, y_raw, window=500.0)
        # iter 0 = initial allocation of k neural Gaussians per anchor (Fig. 9 convention)
        it0 = np.concatenate([[0.0], it])
        y0 = np.concatenate([[float(k)], y])
        mx, my = sample_marks(it0, y0, MARK_ITERS)
        psnr_s = f"{psnr:.2f}" if np.isfinite(psnr) else "—"
        label = f"k={k} (PSNR: {psnr_s})"
        ax.plot(mx, my, color=COLORS[k], lw=2.15, zorder=2)
        ax.plot(
            mx,
            my,
            "o",
            color=COLORS[k],
            ms=8.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
            label=label,
        )
        curves[k] = {
            "run": str(run),
            "psnr": psnr,
            "final": float(my[-1]) if len(my) else None,
        }

    ax.set_xlim(0, 30000)
    ax.set_ylim(0, 21)
    ax.set_yticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    xticks = [0, 5000, 10000, 15000, 20000, 25000, 30000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t // 1000}k" if t else "0" for t in xticks])
    ax.set_ylabel("Activated neural Gaussians", fontweight="bold")
    ax.set_xlabel("Training iterations", fontweight="bold")
    ax.tick_params(axis="x", length=4, pad=4)
    ax.tick_params(axis="y", length=4, pad=3)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 1.02),
        ncol=3,
        frameon=False,
        handlelength=1.15,
        columnspacing=1.35,
        handletextpad=0.35,
        borderaxespad=0.0,
        markerscale=0.95,
    )

    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / "k_ablation.png"
    pdf = fig_dir / "k_ablation.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    routing = OUT_DIR / "routing"
    routing.mkdir(parents=True, exist_ok=True)
    fig.savefig(routing / "k_ablation.png")
    plt.close(fig)

    meta = OUT_DIR / "routing" / "k_ablation.json"
    with open(meta, "w") as f:
        json.dump({"scene": SCENE, "curves": curves, "png": str(png)}, f, indent=2)
    print(f"wrote {png}")
    print(json.dumps(curves, indent=2))


if __name__ == "__main__":
    main()
