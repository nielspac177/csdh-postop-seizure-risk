"""Task 34 — Graphical abstract / pipeline overview figure.

A single horizontally-laid-out schematic showing the full analytical
pipeline: data sources → modelling → conformal decision support →
cost-effectiveness → value-of-information. Headline numbers are
annotated inline. Native matplotlib, no AI generation required.

Output: figures/F0_graphical_abstract.{png,pdf}  +  github_repo/figures/
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from _shared import FIG, OUT

REPO_FIG = OUT / "github_repo" / "figures"
REPO_FIG.mkdir(parents=True, exist_ok=True)

COL = {
    "navy":    "#1F3D5C",
    "rust":    "#B5532C",
    "forest":  "#2E6B45",
    "ochre":   "#B58A2E",
    "soft":    "#F4EFE6",
    "grey":    "#6E6E6E",
    "ink":     "#262320",
}

# Pipeline stages: (title, headline metric, supporting line, fill, outline)
STAGES = [
    {
        "title": "1. Multi-database\ncohort assembly",
        "metric": "n = 655 + 5,376\n+ 218,244",
        "sub":   "BIDMC dev · eICU CRD external\nNIS population",
        "fill":  COL["soft"], "outline": COL["navy"],
    },
    {
        "title": "2. Calibrated\nrisk score",
        "metric": "Firth LR\nAUC 0.681",
        "sub":   "3.3× better calibration\n(Brier 0.069 vs 0.228)",
        "fill":  COL["soft"], "outline": COL["navy"],
    },
    {
        "title": "3. Conformal\ndecision support",
        "metric": "27% rule-out\n11% rule-in",
        "sub":   "Class-conditional Mondrian\n90% coverage at α = 0.10",
        "fill":  COL["soft"], "outline": COL["forest"],
    },
    {
        "title": "4. Cost-effectiveness\nanalysis",
        "metric": "ML-AED is\ndominant",
        "sub":   "Cheaper & more QALYs\nthan observation",
        "fill":  COL["soft"], "outline": COL["rust"],
    },
    {
        "title": "5. Value-of-\ninformation",
        "metric": "$190 M\npopulation EVPI",
        "sub":   "Research-priority frontier:\ncEEG cost · prevalence · AED RRR",
        "fill":  COL["soft"], "outline": COL["ochre"],
    },
]


def draw_stage(ax, x, y, w, h, stage, panel_label):
    # Outer card
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04,rounding_size=0.10",
                          facecolor=stage["fill"], edgecolor=stage["outline"],
                          linewidth=1.6, zorder=2)
    ax.add_patch(box)
    # Top stripe (slightly thinner)
    stripe_h = 0.34
    stripe = FancyBboxPatch((x - w/2, y + h/2 - stripe_h), w, stripe_h,
                             boxstyle="round,pad=0,rounding_size=0.10",
                             facecolor=stage["outline"], edgecolor="none", zorder=3)
    ax.add_patch(stripe)
    # Panel letter in a small white circle on the left of the stripe
    badge_r = 0.10
    badge_cx = x - w/2 + 0.22
    badge_cy = y + h/2 - stripe_h/2
    circ = mpatches.Circle((badge_cx, badge_cy), badge_r,
                            facecolor="white", edgecolor="none", zorder=4)
    ax.add_patch(circ)
    ax.text(badge_cx, badge_cy, panel_label,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=stage["outline"], zorder=5)
    # Title in the stripe (centred, offset slightly right of badge)
    ax.text(x + 0.10, badge_cy, stage["title"],
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", zorder=4, linespacing=1.15)
    # Headline metric (large, color-matched to stage outline)
    ax.text(x, y - 0.02, stage["metric"],
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=stage["outline"], zorder=4, linespacing=1.15)
    # Supporting line (smaller, ink-coloured)
    ax.text(x, y - h/2 + 0.38, stage["sub"],
            ha="center", va="center", fontsize=8.2,
            color=COL["ink"], zorder=4, linespacing=1.4)


def draw_arrow(ax, x0, x1, y, color):
    ar = FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                          mutation_scale=18, color=color, lw=1.8, zorder=1)
    ax.add_patch(ar)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(15, 5.0))
    ax.set_xlim(0, 15); ax.set_ylim(0, 5)
    ax.axis("off")

    # Top title strip
    ax.text(0.4, 4.6,
             "Calibrated, conformally-deployable risk score for postoperative seizure after chronic subdural haematoma evacuation",
             ha="left", va="center", fontsize=13, fontweight="bold",
             color=COL["navy"])
    ax.text(0.4, 4.22,
             "A proof-of-concept multi-database study with value-of-information analysis",
             ha="left", va="center", fontsize=10.5, style="italic", color=COL["grey"])

    # 5 cards across
    n = len(STAGES)
    w, h = 2.55, 1.80
    span = 14.2; left = 0.4
    centres = [left + w/2 + i * (span - w) / (n - 1) for i in range(n)]
    cy = 2.10
    panel_labels = ["A","B","C","D","E"]
    for i, s in enumerate(STAGES):
        draw_stage(ax, centres[i], cy, w, h, s, panel_labels[i])
    # Arrows
    for i in range(n - 1):
        draw_arrow(ax, centres[i] + w/2 + 0.04, centres[i+1] - w/2 - 0.04, cy,
                    color=COL["grey"])

    # Bottom footer band
    ax.text(7.5, 0.45,
             "Twenty-eight reproducible scripts · WCAG 2.2 AA interactive companion site · MIT licensed\n"
             "github.com/nielspac177/csdh-postop-seizure-risk",
             ha="center", va="center", fontsize=9, color=COL["grey"],
             linespacing=1.5)

    plt.tight_layout()
    for d in (FIG, REPO_FIG):
        plt.savefig(d / "F0_graphical_abstract.png", dpi=300, bbox_inches="tight")
        plt.savefig(d / "F0_graphical_abstract.pdf", bbox_inches="tight")
    plt.close()
    print(f"[OK] {FIG / 'F0_graphical_abstract.png'}")
    print(f"     also written to {REPO_FIG / 'F0_graphical_abstract.png'}")


if __name__ == "__main__":
    main()
