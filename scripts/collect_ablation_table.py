#!/usr/bin/env python3
"""Collect component ablation metrics into CSV + markdown summary."""
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

VARIANTS = [
    ("w/o MoE", "baseline"),
    ("w/o tree", "moe"),
    ("w/o CL", "ablate_nocl"),
    ("w/o L_load", "ablate_noload"),
    ("w/o warm-start", "ablate_nowarm"),
    ("Full", "moe_tree"),
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


def main():
    rows = []
    for ds, scenes in SUITES.items():
        for sc in scenes:
            scene = f"{ds}/{sc}"
            rec = {"dataset": ds, "scene": sc}
            for vname, folder in VARIANTS:
                m = latest_metrics(scene, folder)
                if m:
                    rec[f"{vname}_PSNR"] = m["PSNR"]
                    rec[f"{vname}_SSIM"] = m["SSIM"]
                    rec[f"{vname}_LPIPS"] = m["LPIPS"]
                    rec[f"{vname}_run"] = m["run"]
                else:
                    rec[f"{vname}_PSNR"] = ""
                    rec[f"{vname}_SSIM"] = ""
                    rec[f"{vname}_LPIPS"] = ""
                    rec[f"{vname}_run"] = ""
            rows.append(rec)

    csv_path = OUT / "ablations_all_scenes.csv"
    fields = ["dataset", "scene"]
    for vname, _ in VARIANTS:
        fields += [f"{vname}_PSNR", f"{vname}_SSIM", f"{vname}_LPIPS", f"{vname}_run"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # completion + means
    lines = [
        "# Component ablation — all scenes",
        "",
        f"CSV: [ablations_all_scenes.csv](ablations_all_scenes.csv)",
        "",
        "## Completion",
        "",
        "| Variant | Done |",
        "|---|---:|",
    ]
    for vname, folder in VARIANTS:
        n = sum(1 for r in rows if r.get(f"{vname}_PSNR") != "")
        lines.append(f"| {vname} | {n}/{len(rows)} |")

    lines += ["", "## Mean PSNR by dataset", "", "| Dataset | " + " | ".join(v for v, _ in VARIANTS) + " |", "|---|" + "|".join(["---:"] * len(VARIANTS)) + "|"]

    for ds in SUITES:
        sub = [r for r in rows if r["dataset"] == ds]
        cols = []
        for vname, _ in VARIANTS:
            vals = [r[f"{vname}_PSNR"] for r in sub if r.get(f"{vname}_PSNR") != ""]
            if vals:
                cols.append(f"{sum(float(x) for x in vals)/len(vals):.3f}")
            else:
                cols.append("-")
        lines.append(f"| {ds} | " + " | ".join(cols) + " |")

    # overall mean
    cols = []
    for vname, _ in VARIANTS:
        vals = [r[f"{vname}_PSNR"] for r in rows if r.get(f"{vname}_PSNR") != ""]
        cols.append(f"{sum(float(x) for x in vals)/len(vals):.3f}" if vals else "-")
    lines.append(f"| **all** | " + " | ".join(cols) + " |")

    lines += [
        "",
        "## Run missing component ablations",
        "",
        "```bash",
        "bash sweep.sh --suite ablate-components --ablate-components nocl,noload,nowarm --skip-done -g 0",
        "```",
        "",
        "Regenerate this table:",
        "",
        "```bash",
        "python scripts/collect_ablation_table.py",
        "```",
    ]

    md_path = OUT / "ablations_all_scenes.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    for vname, _ in VARIANTS:
        n = sum(1 for r in rows if r.get(f"{vname}_PSNR") != "")
        print(f"  {vname}: {n}/{len(rows)}")


if __name__ == "__main__":
    main()
