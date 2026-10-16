#!/usr/bin/env python3
"""Detailed top-down tree growth figure — every branch shown, no ellipsis."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "documents" / "method_summary" / "routing"

# Full tree texts (every node listed, top → bottom)
FIGURES: dict[str, str] = {}


def _register(name: str, title: str, body: str):
    FIGURES[name] = f"{title}\n{'=' * len(title)}\n\n{body.rstrip()}\n"


_register(
    "A_start",
    "(A) Start — 8 leaves under root (flat MoE warm-start)",
    """
                              root
                               |
       +-------+-------+-------+-------+-------+-------+-------+-------+
       |       |       |       |       |       |       |       |       |
      L0      L1      L2      L3      L4      L5      L6      L7
       o       o       o       o       o       o       o       o

  root
  +-- L0  (leaf expert)
  +-- L1  (leaf expert)
  +-- L2  (leaf expert)
  +-- L3  (leaf expert)
  +-- L4  (leaf expert)
  +-- L5  (leaf expert)
  +-- L6  (leaf expert)
  +-- L7  (leaf expert)
""",
)

_register(
    "B_one_split",
    "(B) One grow — L6 splits downward; L0-L5, L7 unchanged",
    """
                              root
                               |
       +-------+-------+-------+-------+-------+-------+-------+-------+
       |       |       |       |       |       |       |       |       |
      L0      L1      L2      L3      L4      L5      L6      L7
       o       o       o       o       o       o       R       o
                                                       |
                                                  +----+----+
                                                  |         |
                                                 L6a       L6b
                                                  o         o

  root
  +-- L0  (leaf)
  +-- L1  (leaf)
  +-- L2  (leaf)
  +-- L3  (leaf)
  +-- L4  (leaf)
  +-- L5  (leaf)
  +-- L6  (router)  <-- split this step
  |   +-- L6a (leaf)
  |   +-- L6b (leaf)
  +-- L7  (leaf)

  8 leaves -> 9 leaves
""",
)

_register(
    "C_two_top_splits",
    "(C) Two top-level splits — L0 and L6 both expanded",
    """
                              root
                               |
       +-------+-------+-------+-------+-------+-------+-------+-------+
       |       |       |       |       |       |       |       |       |
      L0      L1      L2      L3      L4      L5      L6      L7
       R       o       o       o       o       o       R       o
       |                                       |
  +----+----+                             +----+----+
  |         |                             |         |
 L0a       L0b                           L6a       L6b
  o         o                             o         o

  root
  +-- L0  (router)
  |   +-- L0a (leaf)
  |   +-- L0b (leaf)
  +-- L1  (leaf)
  +-- L2  (leaf)
  +-- L3  (leaf)
  +-- L4  (leaf)
  +-- L5  (leaf)
  +-- L6  (router)
  |   +-- L6a (leaf)
  |   +-- L6b (leaf)
  +-- L7  (leaf)

  8 leaves -> 10 leaves
""",
)

_register(
    "D_subexpert_split",
    "(D) Sub-expert grow — child L0b splits; all 8 top slots still visible",
    """
                              root
                               |
       +-------+-------+-------+-------+-------+-------+-------+-------+
       |       |       |       |       |       |       |       |       |
      L0      L1      L2      L3      L4      L5      L6      L7
       R       o       o       o       o       o       R       o
       |                                       |
  +----+----+                             +----+----+
  |         |                             |         |
 L0a       L0b                           L6a       L6b
  o         R                             o         o
            |
       +----+----+
       |         |
     L0b1       L0b2
       o         o

  root
  +-- L0  (router)
  |   +-- L0a (leaf)
  |   +-- L0b (router)  <-- sub-expert split (depth-2 leaf)
  |       +-- L0b1 (leaf)
  |       +-- L0b2 (leaf)
  +-- L1  (leaf)
  +-- L2  (leaf)
  +-- L3  (leaf)
  +-- L4  (leaf)
  +-- L5  (leaf)
  +-- L6  (router)
  |   +-- L6a (leaf)
  |   +-- L6b (leaf)
  +-- L7  (leaf)

  10 leaves -> 11 leaves
  (kitchen-like: L0_1_c1_13 split at iter 18k)
""",
)

_register(
    "E_multi_head",
    "(E) Same grow check — TWO heads (separate trees, both fully drawn)",
    """
  COLOR head @ iter 6000                 OPACITY head @ iter 6000
  (splits L6)                            (splits L0)

                root                               root
                 |                                  |
  +---+---+---+---+---+---+---+---+    +---+---+---+---+---+---+---+---+
  L0 L1 L2 L3 L4 L5 L6 L7            L0 L1 L2 L3 L4 L5 L6 L7
  o  o  o  o  o  o  R  o             R  o  o  o  o  o  o  o
                    |                |
               +----+----+      +----+----+
              L6a      L6b     L0a      L0b

  COV head @ same iter: no split — all 8 leaves stay single:

                root
                 |
  +---+---+---+---+---+---+---+---+
  L0 L1 L2 L3 L4 L5 L6 L7
  o  o  o  o  o  o  o  o

  Limit: max 2 heads grow per check (here: color + opacity).
""",
)

_register(
    "F_depth_cap",
    "(F) Depth limit — depth-3 leaves cannot split again (moe_tree_depth=3)",
    """
  root
  +-- L0 (router)
  |   +-- L0a (leaf, depth=2)
  |   +-- L0b (router, depth=2)
  |       +-- L0b1 (leaf, depth=3)  <- MAX DEPTH, cannot split
  |       +-- L0b2 (leaf, depth=3)  <- MAX DEPTH, cannot split
  +-- L1 (leaf)
  +-- L2 (leaf)
  +-- L3 (leaf)
  +-- L4 (leaf)
  +-- L5 (leaf)
  +-- L6 (router)
  |   +-- L6a (leaf)
  |   +-- L6b (leaf)
  +-- L7 (leaf)
""",
)


def write_txt(path: Path):
    parts = [
        "# Dynamic tree growth — detailed figures (no omitted branches)",
        "",
        "Every figure lists ALL 8 top-level slots under root.",
        "o = leaf expert    R = router    Split flows TOP -> DOWN",
        "",
    ]
    for key in ["A_start", "B_one_split", "C_two_top_splits", "D_subexpert_split", "E_multi_head", "F_depth_cap"]:
        parts.append(FIGURES[key])
        parts.append("")
    path.write_text("\n".join(parts))


def write_png(path: Path, body: str):
    """Single tall PNG with all detailed ASCII panels."""
    plt.rcParams.update({"font.family": "DejaVu Sans Mono", "savefig.dpi": 160, "savefig.facecolor": "white"})
    fig_h = max(14, 0.012 * len(body.splitlines()))
    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    ax.axis("off")
    ax.text(0.02, 0.99, body, transform=ax.transAxes, va="top", ha="left", fontsize=6.8, linespacing=1.22)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    txt = OUT / "tree_growth_concept_detailed.txt"
    png = OUT / "tree_growth_concept_detailed.png"
    write_txt(txt)
    body = txt.read_text()
    write_png(png, body)
    print(f"wrote {txt}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
