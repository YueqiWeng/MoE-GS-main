#!/usr/bin/env python3
"""5x4 qualitative grid: one red box + corner zoom per panel (church-window style)."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/qiqi/Documents/Scaffold-GS-main")
OUT = ROOT / "documents" / "method_summary"
METHODS = ("3dgs", "baseline", "moe", "moe_tree")
COL_LABELS = ("3D-GS", "Scaffold-GS", "flat MoE", "Tree MoE")
RED = (220, 32, 32)

# Failures that stay obvious at paper size:
# kitchen = see-through melted cabin, bonsai = filled slat gaps,
# playroom = ceiling-light blob, garden = sparkle/fog on the facade,
# truck = dual wheel melted into one + unreadable mudflap.
SCENES = [
    {
        "name": "Kitchen",
        "dataset": "mipnerf360/kitchen",
        "view": "00021.png",
        "box": (820, 160, 200),
        "inset": "bl",
    },
    {
        "name": "Bonsai",
        "dataset": "mipnerf360/bonsai",
        "view": "00005.png",
        "box": (670, 565, 155),
        "inset": "br",
        "inset_frac": 0.50,
    },
    {
        "name": "Playroom",
        "dataset": "db/playroom",
        "view": "00006.png",
        "box": (720, 40, 260),
        "inset": "bl",
    },
    {
        "name": "Garden",
        "dataset": "mipnerf360/garden",
        "view": "00001.png",
        "box": (520, 0, 200),
        "inset": "bl",
    },
    {
        "name": "Truck",
        "dataset": "tandt/truck",
        "view": "00006.png",
        "box": (470, 210, 200),
        "inset": "bl",
    },
]


def latest_run(scene: str, method: str) -> Path:
    d = ROOT / "outputs" / scene / method
    runs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "results.json").exists())
    if not runs:
        raise FileNotFoundError(f"no finished run for {scene}/{method}")
    return runs[-1]


def render_path(scene: str, method: str, view: str) -> Path:
    run = latest_run(scene, method)
    p = run / "test" / "ours_30000" / "renders" / view
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_psnr(scene: str, method: str) -> float:
    with open(latest_run(scene, method) / "results.json") as f:
        return float(json.load(f)["ours_30000"]["PSNR"])


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def add_box_and_inset(
    im: Image.Image, box: tuple[int, int, int], corner: str, panel_w: int, inset_frac: float = 0.42
) -> Image.Image:
    ow, oh = im.size
    panel_h = int(round(panel_w * oh / ow))
    scaled = im.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    sx, sy = panel_w / ow, panel_h / oh
    x, y, s = box
    x = max(0, min(x, ow - s))
    y = max(0, min(y, oh - s))

    inset = int(round(panel_w * inset_frac))
    crop = im.crop((x, y, x + s, y + s)).resize((inset, inset), Image.Resampling.LANCZOS)
    border = max(3, int(round(panel_w * 0.008)))
    pad = max(6, int(round(panel_w * 0.016)))
    tile = Image.new("RGB", (inset + 2 * border, inset + 2 * border), RED)
    tile.paste(crop, (border, border))

    d = ImageDraw.Draw(scaled)
    bx0, by0 = x * sx, y * sy
    bx1, by1 = (x + s) * sx, (y + s) * sy
    d.rectangle([bx0, by0, bx1, by1], outline=RED, width=border)

    tw, th = tile.size
    if corner == "bl":
        px, py = pad, panel_h - th - pad
    elif corner == "br":
        px, py = panel_w - tw - pad, panel_h - th - pad
    elif corner == "tl":
        px, py = pad, pad
    else:  # tr
        px, py = panel_w - tw - pad, pad
    scaled.paste(tile, (int(px), int(py)))
    return scaled


def header_cell(text: str, width: int, height: int) -> Image.Image:
    cell = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(cell)
    f = font(22, bold=True)
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((width - tw) / 2, (height - th) / 2 - 2), text, fill=(20, 20, 20), font=f)
    return cell


def row_label(text: str, width: int, height: int) -> Image.Image:
    cell = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(cell)
    f = font(18, bold=True)
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((width - tw) / 2, (height - th) / 2 - 1), text, fill=(20, 20, 20), font=f)
    return cell


def psnr_bar(psnr: float, width: int, height: int = 24) -> Image.Image:
    bar = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(bar)
    f = font(16, bold=True)
    text = f"{psnr:.2f} dB"
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((width - tw) / 2, (height - th) / 2 - 1), text, fill=(20, 20, 20), font=f)
    return bar


def main() -> None:
    panel_w = 560
    gap = 8
    head_h = 36
    psnr_h = 24
    label_w = 92
    left_gap = 8

    rows_panels = []
    rows_psnr = []
    for spec in SCENES:
        panels = []
        psnrs = []
        for m in METHODS:
            im = Image.open(render_path(spec["dataset"], m, spec["view"])).convert("RGB")
            panels.append(
                add_box_and_inset(im, spec["box"], spec["inset"], panel_w, spec.get("inset_frac", 0.42))
            )
            psnrs.append(load_psnr(spec["dataset"], m))
        rows_panels.append(panels)
        rows_psnr.append(psnrs)

    row_h = [p[0].size[1] + psnr_h for p in rows_panels]
    grid_w = label_w + left_gap + 4 * panel_w + 3 * gap
    grid_h = head_h + gap + sum(row_h) + gap * (len(SCENES) - 1)
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    # column headers
    for i, lab in enumerate(COL_LABELS):
        x = label_w + left_gap + i * (panel_w + gap)
        canvas.paste(header_cell(lab, panel_w, head_h), (x, 0))

    y = head_h + gap
    for spec, panels, psnrs, h in zip(SCENES, rows_panels, rows_psnr, row_h):
        canvas.paste(row_label(spec["name"], label_w, h), (0, y))
        for i, (p, psnr) in enumerate(zip(panels, psnrs)):
            x = label_w + left_gap + i * (panel_w + gap)
            canvas.paste(p, (x, y))
            canvas.paste(psnr_bar(psnr, panel_w, psnr_h), (x, y + p.size[1]))
        y += h + gap

    out_dir = OUT / "figures"
    img_dir = OUT / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "qualitative_5x4.png"
    canvas.save(png, optimize=True)
    canvas.save(img_dir / "qualitative_5x4.png", optimize=True)
    print(f"wrote {png} {canvas.size}")


if __name__ == "__main__":
    main()
