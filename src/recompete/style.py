"""Chart style: navy, teal, blue, and slate only (no red or orange anywhere)."""

from __future__ import annotations

import matplotlib.pyplot as plt

NAVY, TEAL, BLUE, SLATE = "#1e3a5f", "#0f766e", "#2563eb", "#64748b"
STEEL, SEAFOAM, INK, BODY = "#5b8db8", "#5eada5", "#0f172a", "#334155"


def palette() -> list[str]:
    return [NAVY, TEAL, BLUE, SLATE, STEEL, SEAFOAM]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": BODY,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": BODY,
        "ytick.color": BODY,
        "text.color": BODY,
        "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=palette()),
    })  # fmt: skip
