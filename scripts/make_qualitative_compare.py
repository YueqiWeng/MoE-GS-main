#!/usr/bin/env python3
"""Qualitative compares with zooms that differ to the naked eye.

Views/crops were chosen by inspecting test renders, not max-L1 on grass.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "documents" / "method_summary" / "images"

METHODS = ("baseline", "moe", "moe_tree")
LABELS = {"baseline": "baseline", "moe": "moe", "moe_tree": "moe_tree"}
COLORS = [(220, 30, 30), (20, 140, 220), (230, 170, 20)]  # red, blue, gold
PSNR = {
    "bicycle": {"baseline": 25.14, "moe": 25.27, "moe_tree": 25.47},
    "bonsai": {"baseline": 32.52, "moe": 33.07, "moe_tree": 33.75},
    "kitchen": {"baseline": 31.24, "moe": 31.71, "moe_tree": 32.24},
}

# Crop boxes are (x, y, size) in native 1600px test-render coords.
SCENES = {
    "bicycle": {
        "dataset": "mipnerf360/bicycle",
        "view": "00001.png",
        "crops": [
            (520, 20, 300),   # canopy: fog blob vs real trunk
            (40, 80, 300),    # left foliage: smear vs leaves
            (1280, 380, 300), # far brick wall
        ],
    },
    "bonsai": {
        "dataset": "mipnerf360/bonsai",
        "view": "00000.png",
        "crops": [
            (560, 20, 300),   # piano keys: smear vs distinct keys
            (20, 620, 300),   # Honeywell box text
            (0, 270, 280),    # left synth keys
        ],
    },
    "kitchen": {
        "dataset": "mipnerf360/kitchen",
        "view": "00005.png",
        "crops": [
            (0, 0, 300),      # window plants: smear vs leaf structure
            (600, 400, 280),  # track gear: blob vs 8-hole pattern
            (340, 40, 250),   # chair slats (no overlap with plants)
        ],
    },
}


def latest_run(scene: str, method: str) -> Path:
    d = ROOT / "outputs" / scene / method
    runs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "results.json").exists())
    if not runs:
        raise FileNotFoundError(f"no finished run for {scene}/{method}")
    return runs[-1]


def render_path(scene: str, method: str, view: str) -> Path:
    run = latest_run(scene, method)
    rd = next((run / "test").rglob("renders"))
    p = rd / view
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def label_bar(text: str, width: int, height: int = 44) -> Image.Image:
    bar = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(bar)
    f = font(26, bold=True)
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((width - tw) / 2, (height - th) / 2 - 2), text, fill=(20, 20, 20), font=f)
    return bar


def draw_boxes(im: Image.Image, crops: list[tuple[int, int, int]], scale: float) -> Image.Image:
    out = im.copy()
    d = ImageDraw.Draw(out)
    lw = max(4, int(round(5 * scale)))
    for color, (x, y, s) in zip(COLORS, crops):
        x0, y0 = x * scale, y * scale
        x1, y1 = (x + s) * scale, (y + s) * scale
        d.rectangle([x0, y0, x1, y1], outline=color, width=lw)
    return out


def zoom_tile(im: Image.Image, box: tuple[int, int, int], color: tuple[int, int, int], side: int) -> Image.Image:
    x, y, s = box
    crop = im.crop((x, y, x + s, y + s)).resize((side, side), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (side + 10, side + 10), color)
    tile.paste(crop, (5, 5))
    return tile


def caption_row(labels: list[str], width: int) -> Image.Image:
    f = font(16)
    row = Image.new("RGB", (width, 24), (255, 255, 255))
    d = ImageDraw.Draw(row)
    slot = width / len(labels)
    for i, lab in enumerate(labels):
        bbox = d.textbbox((0, 0), lab, font=f)
        tw = bbox[2] - bbox[0]
        d.text((i * slot + (slot - tw) / 2, 2), lab, fill=(70, 70, 70), font=f)
    return row


def compose(name: str) -> Path:
    spec = SCENES[name]
    dataset, view, crops = spec["dataset"], spec["view"], spec["crops"]
    imgs = {
        m: Image.open(render_path(dataset, m, view)).convert("RGB") for m in METHODS
    }
    ow, oh = imgs["baseline"].size
    panel_w = 960
    scale = panel_w / ow
    panel_h = int(round(oh * scale))
    bar_h = 44

    panels = []
    for m in METHODS:
        scaled = imgs[m].resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        scaled = draw_boxes(scaled, crops, scale)
        text = f"{name} | {LABELS[m]} ({PSNR[name][m]:.2f}dB)"
        bar = label_bar(text, panel_w, bar_h)
        stacked = Image.new("RGB", (panel_w, bar_h + panel_h), (255, 255, 255))
        stacked.paste(bar, (0, 0))
        stacked.paste(scaled, (0, bar_h))
        panels.append(stacked)

    gap = 12
    top_w = panel_w * 3 + gap * 2
    top_h = bar_h + panel_h
    top = Image.new("RGB", (top_w, top_h), (255, 255, 255))
    for i, p in enumerate(panels):
        top.paste(p, (i * (panel_w + gap), 0))

    zoom_side = 300
    group_gap = 36
    inner_gap = 8
    group_w = 3 * (zoom_side + 10) + 2 * inner_gap
    zooms = Image.new("RGB", (top_w, zoom_side + 10 + 28), (255, 255, 255))
    # Center the three groups under the three full panels.
    for gi, crop in enumerate(crops):
        gx = gi * (panel_w + gap) + (panel_w - group_w) // 2
        for mi, m in enumerate(METHODS):
            tile = zoom_tile(imgs[m], crop, COLORS[gi], zoom_side)
            zooms.paste(tile, (gx + mi * (zoom_side + 10 + inner_gap), 0))
        cap = caption_row(["baseline", "moe", "moe_tree"], group_w)
        zooms.paste(cap, (gx, zoom_side + 12))

    canvas = Image.new("RGB", (top_w, top.size[1] + 16 + zooms.size[1]), (255, 255, 255))
    canvas.paste(top, (0, 0))
    canvas.paste(zooms, (0, top.size[1] + 16))

    out = OUT / f"{name}__baseline_moe_moe_tree__compare.png"
    canvas.save(out, optimize=True)
    canvas.save(OUT / f"{name}__baseline_moe_moe_tree__compare_framed.png", optimize=True)
    print(f"wrote {out} {canvas.size}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SCENES:
        compose(name)


if __name__ == "__main__":
    main()
