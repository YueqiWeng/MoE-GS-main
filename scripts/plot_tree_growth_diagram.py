#!/usr/bin/env python3
"""Conceptual diagram: how dynamic tree growth works (not statistics)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "documents" / "method_summary" / "routing"
OUT.mkdir(parents=True, exist_ok=True)

# palette
C_ROOT = "#2F6DB5"
C_LEAF = "#54A24B"
C_ROUTER = "#D07A12"
C_SPLIT = "#C43C3C"
C_MUTED = "#BBBBBB"
C_LINE = "#666666"


def draw_root(ax, x, y, w=0.9, h=0.35, label="root router"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=C_ROOT,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)


def draw_leaf(ax, x, y, w=0.55, h=0.28, label="", color=C_LEAF, alpha=1.0, dashed=False):
    ls = (0, (3, 2)) if dashed else "solid"
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color,
        edgecolor="white",
        linewidth=1.0,
        alpha=alpha,
        linestyle=ls,
        zorder=3,
    )
    ax.add_patch(box)
    if label:
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=7.5, fontweight="bold", zorder=4)


def draw_router_small(ax, x, y, w=0.55, h=0.22, label="router"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=C_ROUTER,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=7, fontweight="bold", zorder=4)


def connect(ax, x0, y0, x1, y1, color=C_LINE, lw=1.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-",
            color=color,
            linewidth=lw,
            mutation_scale=1,
            zorder=1,
        )
    )


def panel_flat_moe(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(5, 5.55, "(a) Start: 8 leaves under root (= flat MoE capacity)", ha="center", fontsize=10, fontweight="bold")

    draw_root(ax, 5, 4.6, w=1.4, label="root router\nTop-2 over 8 slots")
    xs = [1.2, 2.3, 3.4, 4.5, 5.5, 6.6, 7.7, 8.8]
    names = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    for x, name in zip(xs, names):
        connect(ax, 5, 4.35, x, 3.35)
        draw_leaf(ax, x, 3.1, label=name)
    ax.text(
        5,
        1.8,
        "Each slot is one expert MLP.\nRouting picks 2 of 8 per anchor.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#DDDDDD"),
    )


def panel_one_split(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(5, 5.55, "(b) One grow step: split ONE busy leaf locally", ha="center", fontsize=10, fontweight="bold")

    draw_root(ax, 5, 4.6, w=1.4, label="root router\nstill 8 top slots")
    xs = [1.2, 2.3, 3.4, 4.5, 5.5, 6.6, 7.7, 8.8]
    names = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    split_x = 6.6
    for x, name in zip(xs, names):
        connect(ax, 5, 4.35, x, 3.55 if x == split_x else 3.35)
        if x == split_x:
            # replaced by sub-router
            draw_router_small(ax, x, 3.35, label="split")
            connect(ax, x, 3.18, x - 0.35, 2.55)
            connect(ax, x, 3.18, x + 0.35, 2.55)
            draw_leaf(ax, x - 0.35, 2.3, label="L6a", color=C_SPLIT)
            draw_leaf(ax, x + 0.35, 2.3, label="L6b", color=C_SPLIT)
            ax.annotate(
                "",
                xy=(x, 2.0),
                xytext=(x + 1.8, 1.15),
                arrowprops=dict(arrowstyle="->", color=C_SPLIT, lw=1.2),
            )
            ax.text(x + 1.85, 1.05, "high load /\nhard leaf", fontsize=7.5, color=C_SPLIT, va="top")
        else:
            draw_leaf(ax, x, 3.1, label=name)

    ax.text(
        5,
        0.55,
        "Root does NOT split into 2 scene groups.\n"
        "Only the chosen leaf becomes router + 2 child experts (+1 leaf total → 9).",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF8F6", edgecolor="#E8C4BC"),
    )


def panel_multi_head(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(5, 5.55, "(c) Three heads → three independent trees", ha="center", fontsize=10, fontweight="bold")

    heads = [
        ("opacity MLP", 1.7, ["L0*", "L1", "L2", "L3", "L4", "L5", "L6", "L7"], 0),
        ("covariance MLP", 5.0, ["L0", "L1*", "L2", "L3", "L4", "L5", "L6", "L7"], 1),
        ("color MLP", 8.3, ["L0", "L1", "L2", "L3", "L4", "L5", "L6*", "L7"], 5),
    ]
    for title, cx, labels, split_idx in heads:
        draw_root(ax, cx, 4.35, w=1.05, h=0.3, label="root")
        lx = [cx - 1.05, cx - 0.75, cx - 0.45, cx - 0.15, cx + 0.15, cx + 0.45, cx + 0.75, cx + 1.05]
        for i, (x, lab) in enumerate(zip(lx, labels)):
            connect(ax, cx, 4.15, x, 3.55, lw=0.8)
            if "*" in lab:
                draw_router_small(ax, x, 3.35, w=0.22, h=0.16, label="×")
                connect(ax, x, 3.22, x - 0.12, 2.85, lw=0.7)
                connect(ax, x, 3.22, x + 0.12, 2.85, lw=0.7)
                draw_leaf(ax, x - 0.12, 2.7, w=0.18, h=0.12, label="", color=C_SPLIT)
                draw_leaf(ax, x + 0.12, 2.7, w=0.18, h=0.12, label="", color=C_SPLIT)
            else:
                draw_leaf(ax, x, 3.35, w=0.22, h=0.16, label="", color=C_LEAF, alpha=0.85)
        ax.text(cx, 4.85, title, ha="center", fontsize=8, fontweight="bold", color="#333333")

    ax.text(
        5,
        1.55,
        "At the same iteration, each head may split a different leaf\n"
        "(or none). Final leaf counts can differ (e.g. 11 / 10 / 11).",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#DDDDDD"),
    )
    ax.text(5, 0.45, "* = leaf selected for split at this grow event", ha="center", fontsize=7.5, color="#777777")


def panel_schedule(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(5, 5.55, "(d) Grow schedule (same for all heads)", ha="center", fontsize=10, fontweight="bold")

    times = [1.0, 3.25, 5.5, 7.75]
    labels = ["warm-start\nfrom flat MoE", "grow check\n6k", "grow check\n10k", "grow check\n14k / 18k"]
    for i, (x, lab) in enumerate(zip(times, labels)):
        c = C_ROOT if i == 0 else C_ROUTER
        draw_leaf(ax, x, 3.8, w=1.5, h=0.9, label=lab, color=c)
        if i < len(times) - 1:
            ax.annotate(
                "",
                xy=(times[i + 1] - 0.85, 3.8),
                xytext=(x + 0.85, 3.8),
                arrowprops=dict(arrowstyle="->", color=C_LINE, lw=1.5),
            )

    # mini evolution strip
    strip_y = 1.8
    ax.text(0.6, strip_y + 0.95, "leaf count (one head):", fontsize=8, color="#555555")
    counts = [("8", 1.2), ("9", 3.0), ("10", 4.8), ("11", 6.6), ("11", 8.4)]
    for txt, x in counts:
        rect = Rectangle((x - 0.35, strip_y - 0.25), 0.7, 0.5, facecolor="#EAF2FB", edgecolor=C_ROOT, lw=0.8)
        ax.add_patch(rect)
        ax.text(x, strip_y, txt, ha="center", va="center", fontsize=9, fontweight="bold", color=C_ROOT)
        if x < 8.0:
            ax.annotate("", xy=(x + 0.55, strip_y), xytext=(x + 0.35, strip_y),
                        arrowprops=dict(arrowstyle="->", color=C_LINE, lw=1.0))

    ax.text(
        5,
        0.55,
        "At most 1 split per head per event (up to max_grows=4).\n"
        "Warm-start copies flat MoE weights 1:1 into 8 leaves before any split.",
        ha="center",
        fontsize=8.5,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F7", edgecolor="#DDDDDD"),
    )


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.facecolor": "white"})

    # 2x2 conceptual figure
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    panel_flat_moe(axes[0, 0])
    panel_one_split(axes[0, 1])
    panel_multi_head(axes[1, 0])
    panel_schedule(axes[1, 1])
    fig.suptitle("Dynamic tree growth (conceptual)", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "tree_growth_concept.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out}")

    # single-panel simplified for inline use
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(8, 4.65, "Tree growth: local split on one leaf (not a root bisection)", ha="center", fontsize=11, fontweight="bold")

    # before
    ax.text(2.5, 4.1, "Before", ha="center", fontsize=9, fontweight="bold", color="#555555")
    draw_root(ax, 2.5, 3.5, w=0.9, label="root")
    for i, x in enumerate([1.0, 1.45, 1.9, 2.35, 2.65, 3.1, 3.55, 4.0]):
        connect(ax, 2.5, 3.28, x, 2.75, lw=0.8)
        color = C_SPLIT if i == 5 else C_LEAF
        draw_leaf(ax, x, 2.55, w=0.28, h=0.22, label="L6" if i == 5 else "", color=color)

    ax.annotate("", xy=(5.8, 2.55), xytext=(4.45, 2.55), arrowprops=dict(arrowstyle="->", lw=2, color=C_LINE))
    ax.text(5.12, 2.85, "grow\none leaf", ha="center", fontsize=8, color=C_SPLIT, fontweight="bold")

    # after
    ax.text(11.5, 4.1, "After", ha="center", fontsize=9, fontweight="bold", color="#555555")
    draw_root(ax, 11.5, 3.5, w=0.9, label="root")
    xs = [10.0, 10.45, 10.9, 11.35, 11.65, 12.1, 12.55, 13.0]
    for i, x in enumerate(xs):
        connect(ax, 11.5, 3.28, x, 2.95 if i == 5 else 2.75, lw=0.8)
        if i == 5:
            draw_router_small(ax, x, 2.75, w=0.35, h=0.18, label="R")
            connect(ax, x, 2.6, x - 0.18, 2.15, lw=0.7)
            connect(ax, x, 2.6, x + 0.18, 2.15, lw=0.7)
            draw_leaf(ax, x - 0.18, 1.95, w=0.22, h=0.16, label="a", color=C_SPLIT)
            draw_leaf(ax, x + 0.18, 1.95, w=0.22, h=0.16, label="b", color=C_SPLIT)
        else:
            draw_leaf(ax, x, 2.55, w=0.28, h=0.22, label="", color=C_LEAF)

    ax.text(8, 0.45, "8 top-level slots unchanged  ·  +1 net leaf  ·  child experts cloned from parent (+ noise)", ha="center", fontsize=8.5, color="#444444")
    out2 = OUT / "tree_growth_concept_simple.png"
    fig.savefig(out2, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
