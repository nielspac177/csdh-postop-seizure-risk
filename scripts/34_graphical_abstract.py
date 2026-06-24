"""Task 34 — Methods figure (study design and analysis workflow).

A single, self-explanatory schematic of WHAT WAS DONE, so a reader grasps the
whole method at a glance: data sources -> risk model -> conformal decision
support -> decision analysis. This is a METHODS figure, not a graphical
abstract: it carries design and procedure only (cohort sizes, model class,
validation scheme, conformal coverage target, PSA design), never results
(no AUCs, no cost-effectiveness verdict, no EVPI value, no conformal yield).

Native matplotlib, no AI generation. Filename kept as F0_graphical_abstract.*
for backward compatibility with the submission/site builders.

Output: figures/F0_graphical_abstract.{png,pdf}  +  github_repo/figures/
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from _shared import FIG, OUT

REPO_FIG = OUT / "github_repo" / "figures"
REPO_FIG.mkdir(parents=True, exist_ok=True)

COL = {
    "navy":   "#1F3D5C",
    "teal":   "#2E6B5E",
    "ochre":  "#B58A2E",
    "rust":   "#B5532C",
    "cream":  "#F6F2EA",
    "grey":   "#6E6E6E",
    "ink":    "#2A2622",
    "line":   "#9AA0A6",
}

X0, X1 = 0.45, 9.55          # card left / right edges


def card(ax, top, bottom, color, step, title):
    """Full-width step card with a coloured header bar."""
    ax.add_patch(FancyBboxPatch(
        (X0, bottom), X1 - X0, top - bottom,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=COL["cream"], edgecolor=color, linewidth=1.7, zorder=2))
    hh = 0.46
    ax.add_patch(FancyBboxPatch(
        (X0, top - hh), X1 - X0, hh,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=color, edgecolor="none", zorder=3))
    ax.text(X0 + 0.30, top - hh / 2, step, ha="left", va="center",
            fontsize=10.5, fontweight="bold", color="white", zorder=4)
    ax.text(X0 + 1.55, top - hh / 2, title, ha="left", va="center",
            fontsize=12.5, fontweight="bold", color="white", zorder=4,
            family="sans-serif")
    return top - hh


def bullet(ax, x, y, text, color, fs=10.3):
    ax.plot([x], [y], marker="s", ms=5.5, color=color, zorder=4,
            markeredgecolor="none")
    ax.text(x + 0.24, y, text, ha="left", va="center", fontsize=fs,
            color=COL["ink"], zorder=4)


def arrow(ax, y_from, y_to):
    ax.add_patch(FancyArrowPatch(
        (5.0, y_from), (5.0, y_to), arrowstyle="-|>",
        mutation_scale=20, color=COL["line"], lw=2.0, zorder=1))


def datasource(ax, x0, x1, top, bottom, color, name, role, lines):
    ax.add_patch(FancyBboxPatch(
        (x0, bottom), x1 - x0, top - bottom,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor="white", edgecolor=color, linewidth=1.3, zorder=4))
    xc = (x0 + x1) / 2
    ax.text(xc, top - 0.30, name, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=color, zorder=5)
    ax.text(xc, top - 0.62, role, ha="center", va="center",
            fontsize=9.0, style="italic", color=COL["grey"], zorder=5)
    y = top - 1.02
    for ln in lines:
        ax.text(xc, y, ln, ha="center", va="center", fontsize=9.8,
                color=COL["ink"], zorder=5)
        y -= 0.34


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(10.5, 12.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.6)
    ax.axis("off")

    # ---- Title -------------------------------------------------------------
    ax.text(X0, 12.18,
            "Predicting postoperative seizure after chronic subdural "
            "haematoma evacuation",
            ha="left", va="center", fontsize=14.5, fontweight="bold",
            color=COL["navy"])
    ax.text(X0, 11.78, "Study design and analysis workflow",
            ha="left", va="center", fontsize=11.5, style="italic",
            color=COL["grey"])

    # ---- STEP 1 · DATA SOURCES --------------------------------------------
    top1, bot1 = 11.45, 8.10
    body = card(ax, top1, bot1, COL["navy"], "STEP 1", "DATA SOURCES")
    ax.text((X0 + X1) / 2, body - 0.30,
            "Outcome:  postoperative seizure after cSDH evacuation",
            ha="center", va="center", fontsize=10.3, fontweight="bold",
            color=COL["navy"])
    ctop, cbot = body - 0.64, bot1 + 0.24
    datasource(ax, 0.85, 4.85, ctop, cbot, COL["navy"],
               "BIDMC", "development cohort (single centre)",
               ["655 cSDH evacuations", "48 postoperative seizures",
                "model development + internal validation"])
    datasource(ax, 5.15, 9.15, ctop, cbot, COL["teal"],
               "eICU-CRD", "external cohort (42 hospitals)",
               ["3,297 subdural-haematoma ICU stays", "300 seizures",
                "external evaluation, related population"])

    arrow(ax, bot1, bot1 - 0.34)

    # ---- STEP 2 · RISK MODEL ----------------------------------------------
    top2, bot2 = bot1 - 0.34, bot1 - 2.50
    body = card(ax, top2, bot2, COL["teal"], "STEP 2", "RISK MODEL")
    bx, y, dy = 0.95, body - 0.34, 0.42
    bullet(ax, bx, y, "Firth penalized logistic regression "
           "(chosen for stable, calibrated estimates at few events)", COL["teal"])
    bullet(ax, bx, y - dy, "Leakage-safe feature set: 18 variables known at "
           "the end of evacuation, before the AED / EEG decision", COL["teal"])
    bullet(ax, bx, y - 2 * dy, "11 model classes compared under nested "
           "cross-validation; Platt recalibration", COL["teal"])
    bullet(ax, bx, y - 3 * dy, "External validation by leave-one-hospital-out "
           "random-effects meta-analysis", COL["teal"])

    arrow(ax, bot2, bot2 - 0.34)

    # ---- STEP 3 · CONFORMAL DECISION SUPPORT ------------------------------
    top3, bot3 = bot2 - 0.34, bot2 - 2.45
    body = card(ax, top3, bot3, COL["ochre"], "STEP 3",
                "CONFORMAL DECISION SUPPORT")
    bx, y, dy = 0.95, body - 0.34, 0.42
    bullet(ax, bx, y, "Class-conditional (Mondrian) split-conformal "
           "prediction", COL["ochre"])
    bullet(ax, bx, y - dy, "90% guaranteed coverage within each class "
           "(α = 0.10)", COL["ochre"])
    bullet(ax, bx, y - 2 * dy, "Each patient receives one of three decision "
           "sets:", COL["ochre"])
    # three pills
    pill_y = y - 3 * dy + 0.02
    pills = [("Rule OUT seizure", COL["teal"]),
             ("Rule IN seizure", COL["rust"]),
             ("Defer  (model abstains)", COL["grey"])]
    px = 1.35
    for label, c in pills:
        w = 0.18 + 0.092 * len(label)
        ax.add_patch(FancyBboxPatch(
            (px, pill_y - 0.17), w, 0.34,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            facecolor="white", edgecolor=c, linewidth=1.5, zorder=4))
        ax.text(px + w / 2, pill_y, label, ha="center", va="center",
                fontsize=9.6, fontweight="bold", color=c, zorder=5)
        px += w + 0.40

    arrow(ax, bot3, bot3 - 0.34)

    # ---- STEP 4 · DECISION ANALYSIS ---------------------------------------
    top4, bot4 = bot3 - 0.34, bot3 - 2.30
    body = card(ax, top4, bot4, COL["rust"], "STEP 4", "DECISION ANALYSIS")
    bx, y, dy = 0.95, body - 0.34, 0.42
    bullet(ax, bx, y, "Cost-effectiveness: decision tree + 10-year Markov "
           "model", COL["rust"])
    bullet(ax, bx, y - dy, "Four management strategies compared by "
           "probabilistic sensitivity analysis (10,000 simulations)", COL["rust"])
    bullet(ax, bx, y - 2 * dy, "Value of information (EVPI / EVPPI) to rank "
           "future research priorities", COL["rust"])

    # ---- Footer ------------------------------------------------------------
    ax.text(5.0, bot4 - 0.45,
            "Fully reproducible analysis  ·  "
            "github.com/nielspac177/csdh-postop-seizure-risk  ·  MIT licensed",
            ha="center", va="center", fontsize=9.2, color=COL["grey"])

    plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
    for d in (FIG, REPO_FIG):
        plt.savefig(d / "F0_graphical_abstract.png", dpi=300, bbox_inches="tight")
        plt.savefig(d / "F0_graphical_abstract.pdf", bbox_inches="tight")
    plt.close()
    print(f"[OK] methods figure -> {FIG / 'F0_graphical_abstract.png'}")
    print(f"     also -> {REPO_FIG / 'F0_graphical_abstract.png'}")


if __name__ == "__main__":
    main()
