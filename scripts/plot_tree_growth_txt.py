#!/usr/bin/env python3
"""ASCII tree growth figures from routing_analysis.json (moe_tree runs)."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "documents" / "method_summary" / "routing" / "tree_growth_txt"

SUITES = {
    "mipnerf360": ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"],
    "tandt": ["train", "truck"],
    "db": ["drjohnson", "playroom"],
    "nerfsynthetic": [
        "chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship",
    ],
}
HEADS = ["opacity", "cov", "color"]
DEFAULT_HEAD = "color"


@dataclass
class TreeNode:
    kind: str  # "leaf" | "node"
    node_id: str
    depth: int
    children: list[TreeNode] = field(default_factory=list)


def strip_prefix(desc: str) -> str:
    desc = desc.strip()
    if " :: " in desc:
        desc = desc.split(" :: ", 1)[1].strip()
    return desc


def parse_tree(desc: str) -> TreeNode:
    desc = strip_prefix(desc)

    def parse_at(i: int) -> tuple[TreeNode, int]:
        if desc.startswith("Leaf(", i):
            j = desc.index(")", i)
            inner = desc[i + 5 : j]
            nid, _, dpart = inner.partition(",d=")
            node = TreeNode("leaf", nid, int(dpart) if dpart else 0)
            return node, j + 1
        if desc.startswith("Node(", i):
            j = desc.index(")", i)
            inner = desc[i + 5 : j]
            nid, _, dpart = inner.partition(",d=")
            if j + 1 >= len(desc) or desc[j + 1] != "[":
                raise ValueError(f"expected '[' after Node(...) at {j}")
            k = j + 2
            children: list[TreeNode] = []
            while k < len(desc):
                while k < len(desc) and desc[k] in ", \t":
                    k += 1
                if k >= len(desc) or desc[k] == "]":
                    break
                child, k = parse_at(k)
                children.append(child)
            if k >= len(desc) or desc[k] != "]":
                raise ValueError("unclosed Node[...]")
            node = TreeNode("node", nid, int(dpart) if dpart else 0, children)
            return node, k + 1
        raise ValueError(f"parse error at pos {i}: {desc[i:i+40]!r}")

    node, end = parse_at(0)
    if end != len(desc):
        raise ValueError(f"trailing junk at {end}: {desc[end:]!r}")
    return node


def leaf_ids_preorder(node: TreeNode) -> list[str]:
    if node.kind == "leaf":
        return [node.node_id]
    out: list[str] = []
    for c in node.children:
        out.extend(leaf_ids_preorder(c))
    return out


def util_map(node: TreeNode, util: list[float] | None) -> dict[str, float]:
    if not util:
        return {}
    ids = leaf_ids_preorder(node)
    n = min(len(ids), len(util))
    return {ids[i]: float(util[i]) for i in range(n)}


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:4.1f}%"


def ascii_lines(
    node: TreeNode,
    umap: dict[str, float],
    *,
    prefix: str = "",
    branch: str = "",
    highlight: set[str] | None = None,
) -> list[str]:
    highlight = highlight or set()
    label = node.node_id
    if node.kind == "leaf":
        tag = " *" if label in highlight else ""
        mass = f"  {fmt_pct(umap[label])}" if label in umap else ""
        return [f"{prefix}{branch}{label}{tag}{mass}"]
    lines = [f"{prefix}{branch}{label}  [router]"]
    ext = prefix + ("    " if branch.startswith("└") else "│   " if branch else "")
    for i, child in enumerate(node.children):
        last = i == len(node.children) - 1
        conn = "└─ " if last else "├─ "
        lines.extend(ascii_lines(child, umap, prefix=ext, branch=conn, highlight=highlight))
    return lines


def parse_grow_event(event: str) -> list[tuple[str, str]]:
    """grow:color:L6_7,opacity:L0_1 -> [(head, leaf_id), ...]"""
    if not str(event).startswith("grow:"):
        return []
    body = str(event)[5:]
    out: list[tuple[str, str]] = []
    for part in body.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        head, lid = part.split(":", 1)
        out.append((head.strip(), lid.strip()))
    return out


def latest_run(dataset: str, scene: str, method: str = "moe_tree") -> Path | None:
    d = ROOT / "outputs" / dataset / scene / method
    if not d.is_dir():
        return None
    runs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "routing_analysis.json").exists())
    return runs[-1] if runs else None


def pick_snapshots(timeline: list[dict]) -> list[dict]:
    """Initial tick, each grow, and final."""
    picks: list[dict] = []
    seen_grow = False
    if timeline:
        picks.append(timeline[0])
    for snap in timeline:
        ev = str(snap.get("event", ""))
        if ev.startswith("grow:"):
            picks.append(snap)
            seen_grow = True
    if timeline and timeline[-1].get("event") == "final":
        if not picks or picks[-1] is not timeline[-1]:
            picks.append(timeline[-1])
    elif timeline and not seen_grow:
        picks.append(timeline[-1])
    return picks


def block_for_snapshot(snap: dict, head: str) -> tuple[str, list[str]]:
    h = (snap.get("heads") or {}).get(head) or {}
    desc = h.get("tree") or ""
    if not desc:
        return "", [f"(no tree snapshot for {head})"]
    node = parse_tree(desc)
    umap = util_map(node, h.get("util"))
    n = int(h.get("n_experts", len(leaf_ids_preorder(node))))
    hist = h.get("growth_history") or []
    grown_now = {lid for hd, lid in parse_grow_event(snap.get("event", "")) if hd == head}
    header = (
        f"iter {snap['iter']:>5}  |  event: {snap.get('event', 'tick')}  |  "
        f"leaves={n}  |  grows={len(hist)}"
    )
    if hist:
        header += f"  |  history: {', '.join(hist)}"
    lines = [header, "-" * min(88, max(40, len(header)))]
    lines.extend(ascii_lines(node, umap, highlight=grown_now))
    return header, lines


def render_scene(
    run_dir: Path,
    dataset: str,
    scene: str,
    head: str,
    *,
    method: str = "moe_tree",
    tag: str = "",
) -> tuple[Path, Path]:
    data = json.loads((run_dir / "routing_analysis.json").read_text())
    timeline = data["timeline"]
    picks = pick_snapshots(timeline)

    method_label = method if method != "moe_tree" else "dynamic tree MoE"
    title = f"{dataset}/{scene}  ·  {head} head  ·  {method_label}"
    txt_lines = [title, "=" * len(title), ""]
    for snap in picks:
        _, block = block_for_snapshot(snap, head)
        txt_lines.extend(block)
        txt_lines.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{scene}__{head}__tree_growth"
    if tag:
        stem = f"{stem}__{tag}"
    txt_path = OUT / f"{stem}.txt"
    png_path = OUT / f"{stem}.png"
    txt_path.write_text("\n".join(txt_lines))

    # Figure: monospace text panel
    fig_h = max(6.0, 0.38 * len(txt_lines))
    fig, ax = plt.subplots(figsize=(11.0, fig_h))
    ax.axis("off")
    body = "\n".join(txt_lines)
    ax.text(
        0.02,
        0.98,
        body,
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="DejaVu Sans Mono",
        fontsize=8.5,
        linespacing=1.25,
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return txt_path, png_path


def write_index(written: list[tuple[str, Path, Path]]) -> None:
    lines = [
        "# Dynamic tree growth (ASCII)",
        "",
        "Text + PNG timelines parsed from `routing_analysis.json` (`heads.<head>.tree`).",
        "Each block: iteration, grow event, leaf count, then an indented tree.",
        "Leaf lines show routing mass (%). `*` = leaf split at that step.",
        "",
    ]
    cur_ds = None
    for ds_scene, txt, png in written:
        ds, scene = ds_scene.split("/", 1)
        if ds != cur_ds:
            lines.append(f"## {ds}")
            cur_ds = ds
        rel_txt = txt.relative_to(OUT)
        rel_png = png.relative_to(OUT)
        lines.append(f"### {scene}")
        lines.append("")
        lines.append(f"<img src=\"{rel_png.as_posix()}\" width=\"920\" alt=\"{scene} tree growth\">")
        lines.append("")
        lines.append(f"Plain text: [{rel_txt.name}]({rel_txt.as_posix()})")
        lines.append("")
    (OUT / "INDEX.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="ASCII dynamic tree growth from routing dumps")
    ap.add_argument("--dataset", default="", help="single dataset, e.g. mipnerf360")
    ap.add_argument("--scene", default="", help="single scene, e.g. kitchen")
    ap.add_argument("--head", default=DEFAULT_HEAD, choices=HEADS)
    ap.add_argument(
        "--method",
        default="moe_tree",
        help="output folder under scene/, e.g. moe_tree or ablate_grows2",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="optional filename suffix, e.g. grows2 → scene__color__tree_growth__grows2",
    )
    ap.add_argument("--all", action="store_true", help="all scenes with matching method runs")
    ap.add_argument(
        "--no-index",
        action="store_true",
        help="do not rewrite INDEX.md (useful for one-off grows=2 dumps)",
    )
    args = ap.parse_args()

    method = args.method
    tag = args.tag
    if method.startswith("ablate_grows") and not tag:
        tag = method.replace("ablate_", "")  # grows2 / grows0 / grows4

    targets: list[tuple[str, str]] = []
    if args.dataset and args.scene:
        targets = [(args.dataset, args.scene)]
    elif args.all:
        for ds, scenes in SUITES.items():
            for scene in scenes:
                if latest_run(ds, scene, method):
                    targets.append((ds, scene))
    else:
        # default demo scenes
        for ds, scene in [
            ("mipnerf360", "kitchen"),
            ("mipnerf360", "bonsai"),
            ("db", "playroom"),
        ]:
            if latest_run(ds, scene, method):
                targets.append((ds, scene))

    written: list[tuple[str, Path, Path]] = []
    for ds, scene in targets:
        run = latest_run(ds, scene, method)
        if run is None:
            print(f"skip {ds}/{scene}: no {method} routing dump")
            continue
        txt, png = render_scene(run, ds, scene, args.head, method=method, tag=tag)
        written.append((f"{ds}/{scene}", txt, png))
        print(f"wrote {txt.relative_to(ROOT)}")
        print(f"wrote {png.relative_to(ROOT)}")

    if written and not args.no_index and method == "moe_tree" and not tag:
        write_index(written)
        print(f"wrote {OUT / 'INDEX.md'}")


if __name__ == "__main__":
    main()
