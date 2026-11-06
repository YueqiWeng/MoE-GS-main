#!/usr/bin/env python3
"""Collect lambda_load sensitivity into CSV + markdown + LaTeX.

Reuses:
  ablate_noload  → lambda_load = 0
  moe_tree       → lambda_load = 0.005 (Full default)
New folders:
  ablate_load_<tag>  e.g. ablate_load_0p001 for 0.001
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "documents" / "method_summary"

SUITES = {
    "mipnerf360": ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"],
    "tandt": ["train", "truck"],
    "db": ["drjohnson", "playroom"],
    "nerfsynthetic": ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"],
}

# (display, folder or None if derived, numeric value)
VALUES = [
    (0.0, "ablate_noload"),
    (0.001, "ablate_load_0p001"),
    (0.005, "moe_tree"),
    (0.01, "ablate_load_0p01"),
    (0.05, "ablate_load_0p05"),
]


def latest_metrics(scene: str, folder: str) -> dict | None:
    d = ROOT / "outputs" / scene / folder
    if not d.is_dir():
        return None
    runs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "results.json").exists())
    if not runs:
        return None
    data = json.loads((runs[-1] / "results.json").read_text())
    block = data.get("ours_30000") or next(iter(data.values()))
    return {
        "PSNR": float(block["PSNR"]),
        "SSIM": float(block["SSIM"]),
        "LPIPS": float(block["LPIPS"]),
        "run": runs[-1].name,
    }


def col(v: float) -> str:
    if v == 0:
        return "0"
    s = f"{v:g}"
    return s


def main() -> None:
    rows = []
    for ds, scenes in SUITES.items():
        for sc in scenes:
            scene = f"{ds}/{sc}"
            rec: dict = {"dataset": ds, "scene": sc}
            for v, folder in VALUES:
                m = latest_metrics(scene, folder)
                key = col(v)
                if m:
                    rec[f"{key}_PSNR"] = m["PSNR"]
                    rec[f"{key}_SSIM"] = m["SSIM"]
                    rec[f"{key}_LPIPS"] = m["LPIPS"]
                    rec[f"{key}_run"] = m["run"]
                else:
                    rec[f"{key}_PSNR"] = ""
                    rec[f"{key}_SSIM"] = ""
                    rec[f"{key}_LPIPS"] = ""
                    rec[f"{key}_run"] = ""
            rows.append(rec)

    keys = [col(v) for v, _ in VALUES]
    csv_path = OUT / "ablate_lambda_load_all_scenes.csv"
    fields = ["dataset", "scene"]
    for k in keys:
        fields += [f"{k}_PSNR", f"{k}_SSIM", f"{k}_LPIPS", f"{k}_run"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    def mean_for(sub: list[dict], k: str) -> float | None:
        vals = [float(r[f"{k}_PSNR"]) for r in sub if r.get(f"{k}_PSNR") != ""]
        return sum(vals) / len(vals) if vals else None

    lines = [
        "# Sensitivity: $\\lambda_{\\mathrm{load}}$",
        "",
        f"CSV: [{csv_path.name}]({csv_path.name})",
        "",
        "Warm-started Tree MoE; only `lambda_load` changes. "
        "`0` = `ablate_noload`, `0.005` = Full `moe_tree`.",
        "",
        "## Completion",
        "",
        "| $\\lambda_{\\mathrm{load}}$ | Done |",
        "|---:|---:|",
    ]
    for k in keys:
        n = sum(1 for r in rows if r.get(f"{k}_PSNR") != "")
        lines.append(f"| {k} | {n}/{len(rows)} |")

    lines += [
        "",
        "## Mean PSNR",
        "",
        "| Split | " + " | ".join(f"$\\lambda={k}$" for k in keys) + " |",
        "|---|" + "|".join(["---:"] * len(keys)) + "|",
    ]
    for ds in SUITES:
        sub = [r for r in rows if r["dataset"] == ds]
        cells = []
        for k in keys:
            m = mean_for(sub, k)
            cells.append(f"{m:.3f}" if m is not None else "-")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    cells = []
    for k in keys:
        m = mean_for(rows, k)
        cells.append(f"{m:.3f}" if m is not None else "-")
    lines.append(f"| **all** | " + " | ".join(cells) + " |")

    md_path = OUT / "ablate_lambda_load_all_scenes.md"
    md_path.write_text("\n".join(lines) + "\n")

    # LaTeX: only complete columns, or all with dash
    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Sensitivity of Tree MoE to the load-balancing weight "
        "$\\lambda_{\\mathrm{load}}$ (warm-started Stage~B; all other knobs fixed). "
        "Mean PSNR on the 19-scene paper split. "
        "$\\lambda{=}0$ disables $\\mathcal{L}_{\\mathrm{load}}$; "
        "$0.005$ is the default.}",
        "\\label{tab:lambda-load}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\small",
        "\\begin{tabular}{l" + "c" * len(keys) + "}",
        "\\toprule",
        "Split & " + " & ".join(f"$\\lambda={k}$" for k in keys) + " \\\\",
        "\\midrule",
    ]
    names = {
        "mipnerf360": "Mip-NeRF 360",
        "tandt": "Tanks \\& Temples",
        "db": "Deep Blending",
        "nerfsynthetic": "NeRF-Synthetic",
    }
    for ds, label in names.items():
        sub = [r for r in rows if r["dataset"] == ds]
        cells = []
        for k in keys:
            m = mean_for(sub, k)
            cells.append(f"{m:.2f}" if m is not None else "--")
        tex.append(f"{label} & " + " & ".join(cells) + " \\\\")
    cells = []
    for k in keys:
        m = mean_for(rows, k)
        cells.append(f"{m:.2f}" if m is not None else "--")
    # bold the default column if complete
    if all(mean_for(rows, k) is not None for k in keys):
        means = [mean_for(rows, k) for k in keys]
        best = max(means)  # type: ignore
        cells = []
        for k in keys:
            m = mean_for(rows, k)
            assert m is not None
            s = f"{m:.2f}"
            if abs(m - best) < 1e-6:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
    tex.append("\\textbf{Paper-19} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex_path = OUT / "table_lambda_load.tex"
    tex_path.write_text("\n".join(tex) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {tex_path}")
    for k in keys:
        n = sum(1 for r in rows if r.get(f"{k}_PSNR") != "")
        m = mean_for(rows, k)
        print(f"  lambda={k}: {n}/{len(rows)}  mean={m if m is None else f'{m:.3f}'}")


if __name__ == "__main__":
    main()
