"""Task 46 — Purpose-built figures for the clinical 15-minute talk.

Clean, large-font, colorblind-safe slide graphics carrying the CURRENT numbers
(the manuscript figures still hold the old conformal/CEA values). Saved to
Manuscript_05192026/clinical_slides/fig/.

Figures:
  sld_conformal.png   — conformal partition donut (rule-out / rule-in / defer)
  sld_cea_curve.png   — incremental NMB (ML-guided − universal AED) vs AED RRR, crossover
  sld_premium.png     — discrimination premium vs random allocation, by RRR
  sld_aed_evidence.png— cSDH AED-efficacy evidence (no significant effect)
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shared import RES, CACHE

OUT = Path("/Users/nielspacheco/Desktop/Research/Ogilvy research/"
           "Data Chronic Subdural Haematoma/Manuscript_05192026/clinical_slides/fig")
OUT.mkdir(parents=True, exist_ok=True)

NAVY, RUST, FOREST, OCHRE, GREY = "#1f3b57", "#b5482a", "#3f6f4f", "#c98a2b", "#8a8a8a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200})


def fig_conformal():
    cf = pd.read_csv(RES / "44_conformal_postopB_firth.csv")
    r = cf[cf.alpha == 0.10].iloc[0]
    ro, ri = r["rule_out_rate"], r["rule_in_rate"]
    defer = 1 - ro - ri
    vals = [ro, ri, defer]
    labels = [f"Rule-OUT seizure\n{ro*100:.0f}%  → observe",
              f"Rule-IN seizure\n{ri*100:.0f}%  → cEEG + AED",
              f"Defer to clinician\n{defer*100:.0f}%"]
    colors = [FOREST, RUST, GREY]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(0.92, 0.5),
              frameon=False, fontsize=14)
    ax.text(0, 0, f"{(ro+ri)*100:.0f}%\nconfident", ha="center", va="center",
            fontsize=20, fontweight="bold", color=NAVY)
    ax.set_title("The model knows when to abstain\n(90% class-conditional coverage, α=0.10)",
                 fontsize=17, fontweight="bold", color=NAVY)
    fig.tight_layout(); fig.savefig(OUT / "sld_conformal.png", bbox_inches="tight"); plt.close(fig)


def fig_cea_curve():
    mr = pd.read_csv(RES / "44_model_vs_random.csv")
    rrr = mr["aed_rrr"].values
    incr = mr["NMB_ml_guided"].values - mr["NMB_universal_aed"].values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color=GREY, lw=1.2, ls="--")
    ax.axvspan(0, 0.30, color=FOREST, alpha=0.12, lw=0)
    ax.plot(rrr, incr, "-o", color=NAVY, lw=3, ms=9, zorder=5)
    ax.annotate("ML-guided preferred", (0.02, max(incr) * 0.7), color=FOREST, fontsize=14, fontweight="bold")
    ax.annotate("Universal AED\npreferred", (0.43, min(incr) * 0.9), color=RUST, fontsize=13,
                fontweight="bold", ha="right")
    ax.text(0.15, ax.get_ylim()[1]*0.05, "cSDH-plausible\n(no proven AED effect)", ha="center",
            fontsize=12, color=FOREST)
    ax.set_xlabel("Assumed AED relative-risk reduction"); ax.set_ylabel("Net benefit:\nML-guided − universal AED ($/patient)")
    ax.set_title("Which strategy wins depends on whether AED works",
                 fontsize=17, fontweight="bold", color=NAVY)
    fig.tight_layout(); fig.savefig(OUT / "sld_cea_curve.png", bbox_inches="tight"); plt.close(fig)


def fig_premium():
    mr = pd.read_csv(RES / "44_model_vs_random.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(mr))
    ax.bar(x, mr["discrimination_premium"], color=NAVY, width=0.6)
    for xi, v in zip(x, mr["discrimination_premium"]):
        ax.text(xi, v + 30, f"${v:,.0f}", ha="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels([f"RRR={r}" for r in mr["aed_rrr"]])
    ax.set_ylabel("Discrimination premium\n($/patient vs random)")
    ax.set_title("The model adds value beyond treating fewer patients\n"
                 "(ML-guided vs random allocation at the same treated fraction)",
                 fontsize=15.5, fontweight="bold", color=NAVY)
    ax.set_ylim(0, max(mr["discrimination_premium"]) * 1.25)
    fig.tight_layout(); fig.savefig(OUT / "sld_premium.png", bbox_inches="tight"); plt.close(fig)


def fig_aed_evidence():
    """Evidence that AED prophylaxis after cSDH is unproven. Each row labelled; only the
    one extractable pooled OR is plotted (with CI), the rest annotated explicitly so the
    panel never looks like a broken/empty forest plot."""
    rows = [
        ("Pacheco-Barrios 2024  ·  meta-analysis, 4,966 pts", 2.62, 0.53, 13.06, None),
        ("Nachiappan & Garg 2021  ·  meta-analysis, 13 studies", None, None, None,
         "no significant reduction"),
        ("Lavergne 2019  ·  adjusted cohort", None, None, None,
         "not protective on multivariable analysis"),
        ("Ratilal (Cochrane)  ·  systematic review", None, None, None,
         "no randomised trial exists"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ys = [3, 2, 1, 0]
    ax.axvline(1.0, color=GREY, ls="--", lw=1.5)
    ax.text(1.0, 4.05, "no effect", color=GREY, fontsize=12, ha="center")
    for y, (lbl, pt, lo, hi, note) in zip(ys, rows):
        ax.text(0.12, y, lbl, va="center", fontsize=12.5, color="#222222")
        if pt is not None:
            ax.errorbar(pt, y - 0.28, xerr=[[pt - lo], [hi - pt]], fmt="s",
                        color=NAVY, ms=11, capsize=5, lw=2.2)
            ax.text(pt, y - 0.55, f"OR {pt} ({lo}–{hi})", ha="center", fontsize=10.5, color=NAVY)
        else:
            ax.text(3.5, y - 0.28, "→ " + note, va="center", fontsize=11, color=RUST, style="italic")
    ax.set_xscale("log"); ax.set_xlim(0.2, 30); ax.set_ylim(-0.8, 4.2)
    ax.set_yticks([]); ax.set_xlabel("Odds ratio for seizure, AED vs none (log scale)")
    ax.set_title("No cSDH study shows AED prophylaxis prevents seizures",
                 fontsize=16.5, fontweight="bold", color=NAVY)
    ax.text(0.22, -0.78, "OR > 1 reflects confounding by indication (sicker haematomas get AED),\n"
            "not causal harm; the honest reading is no proven benefit.",
            fontsize=11, color="#3f4a57")
    fig.tight_layout(); fig.savefig(OUT / "sld_aed_evidence.png", bbox_inches="tight"); plt.close(fig)


def fig_ceiling():
    """Simple, legible replacement for the dense 11-method forest plot: AUC range bar."""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.barh([0], [0.68 - 0.62], left=0.62, height=0.32, color="#cdd6df", edgecolor="none")
    ax.plot([0.645], [0], "D", color=RUST, ms=16)
    ax.text(0.645, 0.32, "Firth (deployed)\nAUC 0.645", ha="center", fontsize=13,
            fontweight="bold", color=RUST)
    ax.annotate("All 11 model families\nland in this band",
                xy=(0.67, 0), xytext=(0.70, -0.35), fontsize=12, color=NAVY,
                ha="left", va="center")
    ax.axvline(0.5, color=GREY, ls=":", lw=1.2); ax.text(0.5, -0.5, "chance", color=GREY,
                                                          ha="center", fontsize=10)
    ax.set_xlim(0.45, 0.85); ax.set_ylim(-0.7, 0.7); ax.set_yticks([])
    ax.set_xlabel("Cross-validated AUC")
    ax.set_title("Discrimination plateaus near 0.68: a sample-size ceiling, not a model failure",
                 fontsize=14.5, fontweight="bold", color=NAVY)
    fig.tight_layout(); fig.savefig(OUT / "sld_ceiling.png", bbox_inches="tight"); plt.close(fig)


def fig_calibration():
    """Clinician-friendly reliability curve for the deployed postop-B model. Honest
    framing: the AVERAGE prediction is right (calibration-in-the-large near 0) but the
    SPREAD is too narrow (slope >1 = under-dispersion), so the curve is steeper than the
    diagonal. The slide says exactly that instead of claiming point-wise agreement."""
    from sklearn.calibration import calibration_curve
    z = np.load(CACHE / "oof_bidmc_postopB_firth.npz")
    y = z["y"].astype(int); p = z["p"].astype(float)
    frac, mean = calibration_curve(y, p, n_bins=6, strategy="quantile")
    base = y.mean()                       # observed base rate (~0.073)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    lim = max(mean.max(), frac.max()) * 1.12
    ax.plot([0, lim], [0, lim], ls=":", color=GREY, lw=1.3, label="Perfect calibration")
    # base-rate crosshair: predictions are pulled toward this line
    ax.axhline(base, color=OCHRE, lw=1.1, ls="--", alpha=0.7)
    ax.text(lim*0.985, base + lim*0.012, f"average risk ≈ {base*100:.0f}%",
            ha="right", va="bottom", fontsize=10.5, color=OCHRE)
    ax.plot(mean, frac, "o-", color=NAVY, lw=2.4, ms=8,
            markeredgecolor="white", markeredgewidth=0.8, label="Deployed model")
    ax.set_xlabel("Predicted seizure risk"); ax.set_ylabel("Observed seizure rate")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title("Average prediction is right; the spread is too narrow",
                 fontsize=15.5, fontweight="bold", color=NAVY)
    ax.annotate("Curve steeper than the line:\nthe model compresses risk\ntoward the average",
                xy=(mean[-1], frac[-1]), xytext=(lim*0.30, lim*0.86),
                fontsize=11.5, color=RUST, va="top",
                arrowprops=dict(arrowstyle="->", color=RUST, lw=1.3))
    ax.legend(loc="lower right", frameon=False, fontsize=11.5)
    fig.tight_layout(); fig.savefig(OUT / "sld_calibration.png", bbox_inches="tight"); plt.close(fig)


def main():
    fig_conformal(); fig_cea_curve(); fig_premium(); fig_aed_evidence(); fig_ceiling()
    fig_calibration()
    print("[OK] slide figures written to", OUT)
    for f in sorted(OUT.glob("sld_*.png")):
        print("   ", f.name)


if __name__ == "__main__":
    main()
