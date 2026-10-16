#!/usr/bin/env python3
"""Measure peak GPU / CPU memory while rendering finished checkpoints."""
from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import resource
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from argparse import ArgumentParser, Namespace  # noqa: E402

from arguments import PipelineParams  # noqa: E402
from gaussian_renderer import prefilter_voxel, render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402


def rss_bytes() -> int:
    # Linux: VmRSS from /proc; fallback to ru_maxrss (kB on Linux).
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def latest_run(method_dir: Path) -> Path | None:
    cands = sorted(
        [p for p in method_dir.glob("*") if p.is_dir() and (p / "cfg_args").exists()],
        reverse=True,
    )
    return cands[0] if cands else None


def load_cfg(model_path: Path) -> dict:
    text = (model_path / "cfg_args").read_text().strip()
    ns = eval(text, {"Namespace": Namespace})  # noqa: S307
    return dict(vars(ns))


def build_gaussians(cfg: dict) -> GaussianModel:
    return GaussianModel(
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


def param_count_mb(gaussians: GaussianModel) -> tuple[int, float]:
    n = 0
    bytes_ = 0
    modules = []
    for name in ("mlp_opacity", "mlp_cov", "mlp_color", "mlp_feature_bank", "embedding_appearance", "anchor_gnn"):
        m = getattr(gaussians, name, None)
        if m is not None and hasattr(m, "parameters"):
            modules.append(m)
    for m in modules:
        for p in m.parameters():
            n += p.numel()
            bytes_ += p.numel() * p.element_size()
    for name in ("_anchor", "_offset", "_anchor_feat", "_scaling", "_rotation", "_opacity"):
        t = getattr(gaussians, name, None)
        if torch.is_tensor(t) and t.numel() > 0:
            n += t.numel()
            bytes_ += t.numel() * t.element_size()
    return n, bytes_ / (1024**2)


def measure_one(model_path: Path, max_views: int = 8) -> dict:
    cfg = load_cfg(model_path)
    cfg["model_path"] = str(model_path)
    # Scene expects source_path / model_path attributes (not _source_path).
    dataset = Namespace(**cfg)
    dataset.model_path = str(model_path)
    dataset.eval = True
    if not hasattr(dataset, "images"):
        dataset.images = "images"
    if not hasattr(dataset, "resolution"):
        dataset.resolution = -1
    if not hasattr(dataset, "data_device"):
        dataset.data_device = "cuda"

    parser = ArgumentParser()
    pipeline = PipelineParams(parser)
    pipe = pipeline.extract(parser.parse_args([]))

    rss0 = rss_bytes()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    gaussians = build_gaussians(cfg)
    scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
    gaussians.eval()
    if gaussians.use_moe:
        gaussians.set_moe_sparse(True)

    n_params, param_mb = param_count_mb(gaussians)
    anchors = int(gaussians.get_anchor.shape[0]) if hasattr(gaussians, "get_anchor") else -1

    bg = torch.tensor(
        [1, 1, 1] if cfg.get("white_background") else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    views = scene.getTestCameras()
    if max_views > 0:
        views = views[:max_views]

    with torch.no_grad():
        for view in views:
            mask = prefilter_voxel(view, gaussians, pipe, bg)
            _ = render(view, gaussians, pipe, bg, visible_mask=mask)

    torch.cuda.synchronize()
    peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_gpu_gb = torch.cuda.max_memory_reserved() / (1024**3)
    rss1 = rss_bytes()
    peak_cpu_gb = max(rss0, rss1) / (1024**3)

    pc_dir = model_path / "point_cloud" / "iteration_30000"
    disk_mb = sum(f.stat().st_size for f in pc_dir.rglob("*") if f.is_file()) / (1024**2)

    del scene, gaussians, views
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "peak_gpu_gb": round(peak_gpu_gb, 3),
        "reserved_gpu_gb": round(reserved_gpu_gb, 3),
        "peak_cpu_rss_gb": round(peak_cpu_gb, 3),
        "params_m": round(n_params / 1e6, 3),
        "param_mb": round(param_mb, 1),
        "anchors": anchors,
        "ckpt_disk_mb": round(disk_mb, 1),
    }


def parse_train_min(log_path: Path) -> float | None:
    text = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text(errors="ignore"))
    times = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    if len(times) < 2:
        return None
    t0 = datetime.strptime(times[0], "%Y-%m-%d %H:%M:%S")
    t1 = datetime.strptime(times[-1], "%Y-%m-%d %H:%M:%S")
    return (t1 - t0).total_seconds() / 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenes",
        default="mipnerf360/kitchen,mipnerf360/bicycle,mipnerf360/bonsai,tandt/truck,db/playroom,nerfsynthetic/lego",
    )
    ap.add_argument("--methods", default="baseline,moe,moe_tree")
    ap.add_argument("--max-views", type=int, default=8)
    ap.add_argument("--out", default="documents/method_summary/compute_cost.csv")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rows = []
    for scene in scenes:
        for method in methods:
            run = latest_run(Path("outputs") / scene / method)
            if run is None:
                print(f"[skip] missing {scene}/{method}")
                continue
            print(f"[meas] {scene} {method} <- {run}")
            try:
                mem = measure_one(run, max_views=args.max_views)
            except Exception as e:
                print(f"[fail] {scene}/{method}: {e}")
                continue
            tmin = parse_train_min(run / "outputs.log") if (run / "outputs.log").exists() else None
            row = {
                "scene": scene,
                "method": method,
                "train_min": round(tmin, 1) if tmin is not None else "",
                **mem,
                "run": str(run),
            }
            rows.append(row)
            print(
                f"  train={row['train_min']}min  "
                f"GPU={row['peak_gpu_gb']}GB  CPU={row['peak_cpu_rss_gb']}GB  "
                f"ckpt={row['ckpt_disk_mb']}MB  anchors={row['anchors']}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[wrote] {out}")

        # means
        print("\n=== mean over measured scenes ===")
        for method in methods:
            rs = [r for r in rows if r["method"] == method]
            if not rs:
                continue
            def avg(k):
                xs = [float(r[k]) for r in rs if r.get(k) not in ("", None)]
                return sum(xs) / len(xs) if xs else float("nan")
            print(
                f"{method:10s}  train={avg('train_min'):5.1f} min  "
                f"GPU={avg('peak_gpu_gb'):5.2f} GB  "
                f"CPU={avg('peak_cpu_rss_gb'):5.2f} GB  "
                f"ckpt={avg('ckpt_disk_mb'):6.1f} MB"
            )


if __name__ == "__main__":
    main()
