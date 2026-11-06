#!/usr/bin/env python3
"""Train official 3D Gaussian Splatting (Kerbl et al., TOG 2023) as a baseline.

This is NOT Scaffold-GS. The original 3D-GS code is cloned into
`third_party/gaussian-splatting` on first run.

Protocol matches existing `baseline` / `moe` / `moe_tree` runs in this repo
(not the 3D-GS paper's images_4 / images_2 recipe):

  images='images', resolution=-1 (cap width at 1600), --eval, 30k iters
  white_background=False (including NeRF-Synthetic, as in existing cfg_args)
  BungeeNeRF: lod=30 split (first 31 cameras after sort), same as sweep.sh
  Metrics: this repo's metrics.py (PSNR / SSIM / LPIPS-VGG) into
           outputs/<scene>/3dgs/<timestamp>/results.json as ours_30000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GS_DIR = ROOT / "third_party" / "gaussian-splatting"
GS_REPO = "https://github.com/graphdeco-inria/gaussian-splatting.git"

BUNGEE_EVAL_PATCH = '''    if eval:
        # Match this project's Scaffold-GS Bungee split (lod=30: first 31 cameras).
        if "bungeenerf" in path:
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx <= 30]
        elif "360" in path:
            llffhold = 8
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        elif llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
'''

ORIG_EVAL_BLOCK = '''    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
'''


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    log("$ " + " ".join(cmd) + (f"  (cwd={cwd})" if cwd else ""))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def patch_renderer_for_local_rasterizer(gs_dir: Path) -> None:
    """This env's diff-gaussian-rasterization is Scaffold-GS vintage:

    - GaussianRasterizationSettings has no `antialiasing`
    - rasterizer returns (color, radii), not (color, radii, depth)
    """
    p = gs_dir / "gaussian_renderer" / "__init__.py"
    text = p.read_text()
    orig = text
    text = text.replace("        antialiasing=pipe.antialiasing\n", "")
    text = text.replace(
        "        rendered_image, radii, depth_image = rasterizer(\n"
        "            means3D = means3D,\n"
        "            means2D = means2D,\n"
        "            shs = shs,\n"
        "            colors_precomp = colors_precomp,\n"
        "            opacities = opacity,\n"
        "            scales = scales,\n"
        "            rotations = rotations,\n"
        "            cov3D_precomp = cov3D_precomp)",
        "        rendered_image, radii = rasterizer(\n"
        "            means3D = means3D,\n"
        "            means2D = means2D,\n"
        "            shs = shs,\n"
        "            colors_precomp = colors_precomp,\n"
        "            opacities = opacity,\n"
        "            scales = scales,\n"
        "            rotations = rotations,\n"
        "            cov3D_precomp = cov3D_precomp)\n"
        "        depth_image = None",
    )
    text = text.replace(
        '"visibility_filter" : (radii > 0).nonzero(),',
        '"visibility_filter" : radii > 0,',
    )
    if text != orig:
        p.write_text(text)
        log("[3dgs] patched gaussian_renderer for Scaffold-GS rasterizer")


def patch_eval_split_for_project(gs_dir: Path) -> None:
    """Bungee in this project uses lod=30, not every-8th COLMAP hold-out."""
    p = gs_dir / "scene" / "dataset_readers.py"
    text = p.read_text()
    if 'if "bungeenerf" in path:' in text and "idx <= 30" in text:
        return
    if ORIG_EVAL_BLOCK not in text:
        raise RuntimeError(f"could not patch 3D-GS eval split in {p}")
    p.write_text(text.replace(ORIG_EVAL_BLOCK, BUNGEE_EVAL_PATCH, 1))
    log("[3dgs] patched BungeeNeRF eval split to lod=30 (match existing runs)")


def patch_blender_uint8(gs_dir: Path) -> None:
    """Newer Pillow rejects np.byte (signed int8) in Image.fromarray.

    Official 3D-GS still uses dtype=np.byte; this env's Pillow 10+ raises
    TypeError: Cannot handle this data type: (1, 1, 3), |i1
    Scaffold-GS already uses np.uint8.
    """
    p = gs_dir / "scene" / "dataset_readers.py"
    text = p.read_text()
    orig = text
    text = text.replace("dtype=np.byte", "dtype=np.uint8")
    if "if not os.path.exists(image_path):" not in text:
        text = text.replace(
            "            image_path = os.path.join(path, cam_name)\n"
            "            image_name = Path(cam_name).stem\n"
            "            image = Image.open(image_path)\n",
            "            image_path = os.path.join(path, cam_name)\n"
            "            image_name = Path(cam_name).stem\n"
            "            if not os.path.exists(image_path):\n"
            "                continue\n"
            "            image = Image.open(image_path)\n",
        )
    if text != orig:
        p.write_text(text)
        log("[3dgs] patched Blender loader (uint8 + skip missing frames)")


def patch_blender_composite(gs_dir: Path) -> None:
    """Newer 3D-GS reloads PNG from disk in loadCam and drops the composited RGB.

    RGBA + alpha_mask makes train.py ignore the background (image *= alpha_mask),
    so Gaussians explode in empty space (synthetic PSNR ~3 dB). Composite onto
    the dataset background here, matching Scaffold-GS (white_background=False).
    """
    p = gs_dir / "utils" / "camera_utils.py"
    text = p.read_text()
    needle = "def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):\n    image = Image.open(cam_info.image_path)\n"
    insert = (
        "def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):\n"
        "    image = Image.open(cam_info.image_path)\n"
        "    if is_nerf_synthetic:\n"
        "        im_data = np.array(image.convert('RGBA')) / 255.0\n"
        "        bg = np.array([1, 1, 1]) if args.white_background else np.array([0, 0, 0])\n"
        "        arr = im_data[:, :, :3] * im_data[:, :, 3:4] + bg * (1 - im_data[:, :, 3:4])\n"
        "        image = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), 'RGB')\n"
    )
    if "if is_nerf_synthetic:" in text and "im_data[:, :, 3:4]" in text:
        return
    if needle not in text:
        raise RuntimeError(f"could not patch Blender composite in {p}")
    p.write_text(text.replace(needle, insert, 1))
    log("[3dgs] patched loadCam to composite Blender RGBA onto background")


def ensure_gaussian_splatting() -> Path:
    if not (GS_DIR / "train.py").exists():
        GS_DIR.parent.mkdir(parents=True, exist_ok=True)
        log(f"[3dgs] cloning {GS_REPO} -> {GS_DIR} (no SIBR viewers)")
        run(["git", "clone", "--filter=blob:none", GS_REPO, str(GS_DIR)])
    patch_renderer_for_local_rasterizer(GS_DIR)
    patch_eval_split_for_project(GS_DIR)
    patch_blender_uint8(GS_DIR)
    patch_blender_composite(GS_DIR)
    return GS_DIR


def data_source_path(scene: str) -> Path:
    p = ROOT / "data" / scene
    if not p.exists():
        raise FileNotFoundError(f"missing dataset: {p}")
    return p


def write_cfg_args(out_dir: Path, source: Path, iterations: int, extra: list[str]) -> None:
    # sweep.sh looks for cfg_args next to results.json
    text = (
        f"Namespace(source_path='{source}', model_path='{out_dir}', "
        f"images='images', resolution=-1, white_background=False, eval=True, "
        f"iterations={iterations}, method='3dgs', extra={extra!r})\n"
    )
    (out_dir / "cfg_args").write_text(text)


def _metric_block(raw):
    if not isinstance(raw, dict):
        return None
    if "PSNR" in raw:
        return raw
    for key in ("ours_30000", "ours"):
        if key in raw and isinstance(raw[key], dict) and "PSNR" in raw[key]:
            return raw[key]
    first = next(iter(raw.values()), None)
    if isinstance(first, dict) and "PSNR" in first:
        return first
    if isinstance(first, dict):
        return _metric_block(first)
    return None


def collect_metrics(out_dir: Path) -> dict:
    src = out_dir / "results.json"
    if not src.exists():
        raise FileNotFoundError(f"metrics.py did not write {src}")
    raw = json.loads(src.read_text())
    block = _metric_block(raw)
    if block is None:
        raise RuntimeError(f"unrecognized results.json: {src}")
    ours = {
        "SSIM": float(block["SSIM"]),
        "PSNR": float(block["PSNR"]),
        "LPIPS": float(block["LPIPS"]),
    }
    (out_dir / "results.json").write_text(json.dumps({"ours_30000": ours}, indent=True) + "\n")
    return ours


def train_and_eval(
    scene: str,
    gpu: str,
    iterations: int,
    logdir: str,
    extra_cli: list[str],
) -> Path:
    ensure_gaussian_splatting()
    source = data_source_path(scene)

    stamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    out_dir = ROOT / "outputs" / scene / logdir / stamp
    gs_model = out_dir / "gs_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    gs_model.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Keep conda env (simple-knn / diff-gaussian-rasterization already present).
    env["PYTHONPATH"] = str(GS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    port = str(random.randint(10000, 30000))
    is_blender = (source / "transforms_train.json").exists()
    is_bungee = scene.startswith("bungeenerf/")
    # Same load flags as existing baseline/moe/moe_tree cfg_args:
    # images=images, resolution=-1, white_background=False, eval=True.
    train_cmd = [
        sys.executable,
        str(GS_DIR / "train.py"),
        "-s",
        str(source),
        "-m",
        str(gs_model),
        "--iterations",
        str(iterations),
        "--eval",
        "--quiet",
        "--disable_viewer",
        "--port",
        port,
        "--test_iterations",
        str(iterations),
        "--save_iterations",
        str(iterations),
    ]
    if not is_blender:
        train_cmd += ["--images", "images"]
    # Bungee: official 3D-GS densifies far more Gaussians than Scaffold-GS and
    # OOM'd at 1600px on this 16GB GPU. Keep the same image size; move cameras
    # off GPU and raise the densify threshold slightly so training fits.
    if is_bungee:
        train_cmd += ["--data_device", "cpu", "--densify_grad_threshold", "0.0004"]
    train_cmd += extra_cli

    write_cfg_args(out_dir, source, iterations, train_cmd)
    log("[method] 3dgs = official 3D Gaussian Splatting (Kerbl et al. 2023)")
    log("[protocol] images=images resolution=-1 white_background=False eval=True")
    if is_bungee:
        log("[bungee] data_device=cpu densify_grad_threshold=0.0004 (avoid 16GB OOM)")
    log(f"[scene]  {scene}")
    log(f"[out]    {out_dir}")
    run(train_cmd, cwd=GS_DIR, env=env)

    render_cmd = [
        sys.executable,
        str(GS_DIR / "render.py"),
        "-m",
        str(gs_model),
        "--skip_train",
        "--quiet",
    ]
    run(render_cmd, cwd=GS_DIR, env=env)

    test_src = gs_model / "test"
    test_dst = out_dir / "test"
    if not test_src.exists():
        raise FileNotFoundError(f"3D-GS render did not write {test_src}")
    if test_dst.exists():
        shutil.rmtree(test_dst)
    shutil.copytree(test_src, test_dst)

    # Same PSNR/SSIM/LPIPS-VGG as baseline/moe/moe_tree (this repo's metrics.py).
    env_metrics = env.copy()
    env_metrics["PYTHONPATH"] = str(ROOT)
    run(
        [sys.executable, str(ROOT / "metrics.py"), "-m", str(out_dir)],
        cwd=ROOT,
        env=env_metrics,
    )

    ours = collect_metrics(out_dir)
    log(
        f"[3dgs] {scene}  PSNR={ours['PSNR']:.3f}  "
        f"SSIM={ours['SSIM']:.4f}  LPIPS={ours['LPIPS']:.4f}"
    )
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Official 3D-GS baseline wrapper")
    ap.add_argument("-d", "--data", required=True, help="dataset/scene, e.g. mipnerf360/kitchen")
    ap.add_argument("-g", "--gpu", default="0")
    ap.add_argument("-l", "--logdir", default="3dgs")
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="extra flags forwarded to 3D-GS train.py")
    args = ap.parse_args()
    extra = [a for a in args.rest if a != "--"]
    train_and_eval(args.data, args.gpu, args.iterations, args.logdir, extra)


if __name__ == "__main__":
    main()
