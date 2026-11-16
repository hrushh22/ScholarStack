"""Shared matplotlib style for every report figure.

Palette: scarlet first, then blue, green, gold, violet (fixed order), one-hue scarlet ramp for magnitude, recessive hairline grid, thin
marks, values labelled selectively. Text never takes a series colour.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import path  # noqa: E402

# Scarlet first (SDSU), then the validated categorical hues; passes colour-blind checks in order.
SERIES = ["#a6192e", "#2a78d6", "#1baf7a", "#eda100", "#6250d6", "#e87ba4", "#008300", "#eb6834"]
SCARLET_RAMP = ["#f6dade", "#edb6bf", "#e1909d", "#d4697c", "#c5435b", "#a6192e", "#8c1426", "#6f0f1e"]
NEUTRAL = "#b9b4ae"   # comparison systems, so ours stands out
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OTHER = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.markersize": 6,
    "figure.dpi": 110, "savefig.dpi": 180, "savefig.bbox": "tight",
})


def savefig(fig, name: str, subtitle: str | None = None) -> None:
    if subtitle:
        fig.text(0.01, -0.02, subtitle, color=MUTED, fontsize=8.5, ha="left", va="top")
    out = path("figures") / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)


def hbar(ax, labels, values, color=SERIES[0], fmt="{:,.0f}", height=0.62):
    """Horizontal bars with the value printed at each tip (used for ranked lists)."""
    y = range(len(labels))
    ax.barh(y, values, color=color, height=height, edgecolor=SURFACE, linewidth=0)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    vmax = max(values) if len(values) else 1
    for yi, v in zip(y, values):
        ax.text(v + vmax * 0.01, yi, fmt.format(v), va="center", ha="left", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, vmax * 1.12)
