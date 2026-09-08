"""Design tokens and figure plumbing.

Charts are built for a printed slide: a light surface, no interaction layer,
and every series direct-labelled so identity never rests on colour alone. The
aqua slot sits below 3:1 contrast on this surface, so labels are required
rather than optional.

Palette: slots 1-3 of a categorical set that clears the all-pairs
colour-vision separation floors. Ordered quantities - the neighbourhood
segments - use a single-hue sequential ramp instead, because their order
carries meaning.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e3e2dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua
NEUTRAL = "#b8b6ae"
CRITICAL = "#e34948"

RC_PARAMS = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
}


def use_house_style() -> None:
    """Apply the shared rcParams. Called once, at the top of the figure stage."""
    plt.rcParams.update(RC_PARAMS)


def style(ax: plt.Axes, *, ygrid: bool = True) -> None:
    """Recessive grid behind the marks, no chartjunk."""
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig: plt.Figure, name: str) -> None:
    """Write a figure as PDF for the slides and PNG for the dashboard."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(config.FIGURES / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote outputs/figures/{name}.pdf")
