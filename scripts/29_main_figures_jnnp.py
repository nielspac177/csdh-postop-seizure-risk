"""Task 29 — Main paper figures, JNNP-style.

JNNP aesthetic conventions (consistent with the journal's published figures):
  • Helvetica / Arial sans-serif throughout (DejaVu Sans fallback)
  • Open-box axes: only bottom and left spines visible; tick direction out
  • Bold uppercase panel labels (A, B, C) in the top-left of each panel
  • Restrained palette: navy / rust / forest-green / muted-grey
  • Light grey y-axis grid only, alpha 0.30
  • 300 dpi PNG (print) and vector PDF (for figure submission)
  • CMYK-safe primary colours; usable in grayscale conversion

Rebuilds F1–F6 as native matplotlib figures (no PIL image-wrapping),
sourced directly from the result CSVs and the existing scripts' data.

Outputs:
  figures/F1_discrimination.{png,pdf}
  figures/F2_calibration_dca.{png,pdf}
  figures/F3_method_battery.{png,pdf}
  figures/F4_conformal.{png,pdf}
  figures/F5_cea.{png,pdf}
  figures/F6_voi.{png,pdf}
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches
import json

from _shared import RES, FIG, CACHE

# ── JNNP style ──────────────────────────────────────────────
# Prefer Helvetica / Arial; fall back to a sans-serif that ships with matplotlib.
_pref_fonts = ["Helvetica", "Helvetica Neue", "Arial", "Liberation Sans",
                "DejaVu Sans"]
_available = {f.name for f in font_manager.fontManager.ttflist}
JNNP_FONT = next((f for f in _pref_fonts if f in _available), "DejaVu Sans")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [JNNP_FONT],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelweight": "regular",
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.45,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.6,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,          # editable text in PDF
    "ps.fonttype": 42,
})

# JNNP-safe palette (CMYK-friendly, distinct in grayscale)
COL = {
    "navy":    "#1F3D5C",
    "rust":    "#B5532C",
    "forest":  "#2E6B45",
    "indigo":  "#5C4B8A",
    "ochre":   "#B58A2E",
    "slate":   "#4A5560",
    "soft":    "#9FB6C9",
    "rule":    "#000000",
    "grey":    "#6E6E6E",
    "highlight":"#D03C29",
}

def add_panel_label(ax, label, *, offset=(-0.12, 1.04)):
    ax.text(offset[0], offset[1], label,
             transform=ax.transAxes,
             fontsize=11, fontweight="bold",
             va="bottom", ha="left")

def style_axis(ax, *, ygrid=True, xgrid=False):
    ax.tick_params(axis="both", which="major")
    if ygrid:
        ax.yaxis.grid(True, color=COL["soft"], alpha=0.30, linewidth=0.4)
    if xgrid:
        ax.xaxis.grid(True, color=COL["soft"], alpha=0.30, linewidth=0.4)
    else:
        ax.xaxis.grid(False)

def figure_legend_below(fig, handles, labels, *, ncol=None, y=0.02, fontsize=8):
    """Place a single shared legend below the figure, centred horizontally.
    Frees the panels of inline legends so they never overlap data."""
    if ncol is None:
        ncol = min(len(handles), 4)
    fig.legend(handles, labels, loc="lower center",
                bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                fontsize=fontsize, handlelength=1.6, handletextpad=0.6,
                columnspacing=1.6)


# ─── F1 — Multi-database discrimination ─────────────────────
def figure_1():
    firth = pd.read_csv(RES / "24_firth_bayes_lr.csv")
    loho  = pd.read_csv(RES / "04_loho_summary.csv")
    leak  = pd.read_csv(RES / "05_leakage_audit.csv")

    fig = plt.figure(figsize=(7.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                            hspace=0.55, wspace=0.45,
                            bottom=0.13, top=0.96, left=0.18, right=0.97)

    # Panel A: BIDMC primary (Firth + BRF, postop_A + B)
    axA = fig.add_subplot(gs[0, 0])
    primary = firth[firth["model"].isin(["BalancedRF (baseline)", "Firth penalized LR"])]
    primary = primary.sort_values(["feature_set", "model"]).reset_index(drop=True)
    pos = np.arange(len(primary))
    colors = [COL["navy"] if "BalancedRF" in m else COL["rust"]
              for m in primary["model"]]
    axA.errorbar(primary["auc"], pos,
                  xerr=[primary["auc"] - primary["ci_lo"],
                        primary["ci_hi"] - primary["auc"]],
                  fmt="none", ecolor=COL["slate"], elinewidth=0.9, capsize=2.5)
    for i, c in enumerate(colors):
        axA.scatter(primary["auc"].iloc[i], pos[i], color=c, s=42, zorder=3,
                     edgecolor="black", linewidth=0.5)
    axA.set_yticks(pos)
    axA.set_yticklabels(
        [f"{r.feature_set} · "
         f"{r.model.replace('penalized', 'pen.').replace(' (baseline)', '')}"
         for r in primary.itertuples()], fontsize=8)
    axA.invert_yaxis()
    axA.axvline(0.5, ls=":", color=COL["grey"], lw=0.7)
    axA.set_xlim(0.45, 0.85)
    axA.set_xlabel("Cross-validated AUC (95% CI)")
    style_axis(axA, xgrid=True, ygrid=False)
    add_panel_label(axA, "A")

    # Panel B: LOHO pooled random-effects
    axB = fig.add_subplot(gs[0, 1])
    if "auc_pooled_RE" in loho.columns:
        loho_p = loho.sort_values(["cohort", "set"]).reset_index(drop=True)
        pos = np.arange(len(loho_p))
        axB.errorbar(loho_p["auc_pooled_RE"], pos,
                      xerr=[loho_p["auc_pooled_RE"] - loho_p["auc_pooled_RE_lo"],
                            loho_p["auc_pooled_RE_hi"] - loho_p["auc_pooled_RE"]],
                      fmt="none", ecolor=COL["slate"], elinewidth=0.9, capsize=2.5)
        for i, row in loho_p.iterrows():
            axB.scatter(row["auc_pooled_RE"], i, marker="D", color=COL["navy"],
                          s=52, zorder=3, edgecolor="black", linewidth=0.5)
        axB.set_yticks(pos)
        labels = [(f"{row['cohort']}/{row['set']}\n"
                    f"I²={row['I2_pct']:.0f}%, k={int(row['n_hospitals'])}")
                    for _, row in loho_p.iterrows()]
        axB.set_yticklabels(labels, fontsize=8)
        axB.invert_yaxis()
        axB.axvline(0.5, ls=":", color=COL["grey"], lw=0.7)
        axB.set_xlim(0.45, 0.85)
        axB.set_xlabel("Random-effects pooled AUC (95% CI)")
        style_axis(axB, xgrid=True, ygrid=False)
        add_panel_label(axB, "B")

    # Panel C: temporal leakage sensitivity
    axC = fig.add_subplot(gs[1, :])
    leak_p = leak[leak["cohort"].str.contains("eICU", case=False, na=False)].copy()
    leak_p = leak_p.dropna(subset=["auc"]).reset_index(drop=True)
    # The three exclusion-window rows all carry spec="Set C original"; the cohort
    # column is what distinguishes them (seizures kept only if >=1h/24h/72h after
    # index). Build informative y-labels so they don't read as duplicates.
    def _panelC_label(row):
        import re
        spec, cohort, ev = str(row["spec"]), str(row["cohort"]), int(row["events"])
        if spec == "Set C original":
            m = re.search(r"(\d+)h", cohort)
            win = m.group(1) if m else "?"
            return f"Exclude seizures <{win}h ({ev} events)"
        if "103 features" in spec:
            return "Full Set C (103 features)"
        if "pre-seizure features only" in spec:
            return "Pre-seizure features only (22)"
        if "0-72h" in spec:
            return "0–72h seizure outcome (strict feats)"
        return spec
    leak_p["plot_label"] = [_panelC_label(r) for _, r in leak_p.iterrows()]
    if len(leak_p) > 0:
        pos = np.arange(len(leak_p))
        axC.errorbar(leak_p["auc"], pos,
                       xerr=[leak_p["auc"] - leak_p["lo"],
                             leak_p["hi"] - leak_p["auc"]],
                       fmt="none", ecolor=COL["slate"], elinewidth=0.9,
                       capsize=2.5)
        for i, row in leak_p.iterrows():
            color = COL["forest"] if "strict" in row["spec"].lower() else COL["navy"]
            axC.scatter(row["auc"], i, color=color, s=42, zorder=3,
                         edgecolor="black", linewidth=0.5)
        axC.set_yticks(pos)
        axC.set_yticklabels(leak_p["plot_label"], fontsize=8)
        axC.invert_yaxis()
        axC.axvline(0.5, ls=":", color=COL["grey"], lw=0.7)
        axC.set_xlim(0.45, 0.85)
        axC.set_xlabel("Cross-validated AUC (95% CI)")
        style_axis(axC, xgrid=True, ygrid=False)
        add_panel_label(axC, "C")

    # Shared legend below the figure, two colour categories + reference line
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker="o", color="w",
                       markerfacecolor=COL["forest"], markeredgecolor="black",
                       markersize=7),
               Line2D([0],[0], marker="o", color="w",
                       markerfacecolor=COL["navy"], markeredgecolor="black",
                       markersize=7),
               Line2D([0],[0], marker="D", color="w",
                       markerfacecolor=COL["navy"], markeredgecolor="black",
                       markersize=7),
               Line2D([0],[0], color=COL["grey"], ls=":", lw=1)]
    labels = ["Strict pre-seizure features (panel C)",
               "Full feature set",
               "Random-effects pooled (panel B)",
               "Chance (AUC = 0.5)"]
    figure_legend_below(fig, handles, labels, ncol=4, y=0.02, fontsize=8)

    plt.savefig(FIG / "F1_discrimination.png")
    plt.savefig(FIG / "F1_discrimination.pdf")
    plt.close()
    print("[OK] F1_discrimination — JNNP style")


# ─── F2 — Calibration + DCA ─────────────────────────────────
def figure_2():
    from statsmodels.nonparametric.smoothers_lowess import lowess
    cal = pd.read_csv(RES / "02_calibration_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    plt.subplots_adjust(wspace=0.34, bottom=0.16, top=0.92,
                          left=0.08, right=0.97)

    # ── Panel A: LOWESS-smoothed calibration curves ──
    # Smooth non-parametric calibration (E[y | p_pred]) of the DEPLOYED Firth
    # model (oof_*_firth), not the BalancedRandomForest / LR-EN+SMOTE comparator
    # caches the paper rejects. Each curve is drawn only over the range of
    # predicted risk the model actually produces (to its 97.5th percentile,
    # capped at the 0.30 axis); a light 95% bootstrap band shows uncertainty and
    # a marginal rug shows predicted-risk density, making the deployed model's
    # under-dispersion (predictions compressed near the base rate) visible.
    axA = axes[0]
    axA.plot([0, 0.5], [0, 0.5], color=COL["grey"], lw=0.9, ls=":",
              label="Perfect calibration", zorder=1)
    # frac = LOWESS smoothing span; the deployed BIDMC model's predictions are
    # squeezed into a very narrow band (~0.065-0.134), so it needs a wider span
    # to read as a clean line rather than a wiggle.
    cal_models = [
        ("eicu_setC",           "eICU Set C",                COL["navy"], "-",  0.60),
        ("bidmc_postopB_firth", "BIDMC postop-B (deployed)", COL["rust"], "--", 0.90),
    ]
    rng = np.random.default_rng(42)
    rug_base = -0.018
    for key, label, color, ls, frac in cal_models:
        cache_path = CACHE / f"oof_{key}.npz"
        if not cache_path.exists():
            continue
        z = np.load(cache_path)
        y = z["y"].astype(float); p = np.clip(z["p"], 1e-6, 1 - 1e-6)
        hi_cap = min(float(np.percentile(p, 97.5)), 0.305)
        grid = np.linspace(float(p.min()), hi_cap, 60)
        smooths = []
        for _ in range(200):
            idx = rng.integers(0, len(y), len(y))
            try:
                sm = lowess(y[idx], p[idx], frac=frac, return_sorted=True,
                              it=0, missing="drop")
            except Exception:
                continue
            if len(sm) < 5: continue
            smooths.append(np.interp(grid, sm[:, 0], sm[:, 1]))
        if not smooths: continue
        smooths = np.clip(np.vstack(smooths), 0, 1)
        lo  = np.percentile(smooths, 2.5,  axis=0)
        hi  = np.percentile(smooths, 97.5, axis=0)
        mid = np.percentile(smooths, 50,   axis=0)
        axA.fill_between(grid, lo, hi, color=color, alpha=0.10, lw=0, zorder=2)
        axA.plot(grid, mid, color=color, lw=2.2, ls=ls, label=label, zorder=3)
        # marginal rug of predicted risk (under the curve)
        rug = p[p <= 0.305]
        axA.plot(rug, np.full(rug.shape, rug_base), marker="|", ls="none",
                  color=color, alpha=0.10, ms=6, zorder=1, clip_on=False)
        rug_base -= 0.020
    axA.set_xlim(0, 0.305); axA.set_ylim(-0.05, 0.45)
    axA.set_xlabel("Predicted probability")
    axA.set_ylabel("Observed event rate")
    axA.set_title("Calibration of the deployed model")
    style_axis(axA, ygrid=True, xgrid=True)
    add_panel_label(axA, "A")
    # Panel-local legend: place inside the panel rather than the shared bottom
    axA.legend(loc="upper left", fontsize=7.5, frameon=False,
                handlelength=2.0, handletextpad=0.5,
                title="(LOWESS · 95% bootstrap band; rug = predicted-risk density)",
                title_fontsize=6.5)
    axA_handles, axA_labels = [], []  # nothing to forward to the shared legend

    # Panel B: decision-curve net benefit, computed from the SAME deployed
    # Firth OOF predictions as panel A (not the stale BRF/LR-EN DCA CSV, whose
    # inflated predictions drove the BIDMC curve sharply negative in the 5-15%
    # band — a calibration artefact, not a true loss of clinical utility).
    axB = axes[1]

    def net_benefit(y, p, thresholds):
        y = y.astype(int); n = len(y); prev = y.mean()
        nb_m, nb_all = [], []
        for t in thresholds:
            pos = p >= t
            tp = np.sum(pos & (y == 1)) / n
            fp = np.sum(pos & (y == 0)) / n
            w = t / (1.0 - t)
            nb_m.append(tp - fp * w)
            nb_all.append(prev - (1.0 - prev) * w)
        return np.array(nb_m), np.array(nb_all)

    thr = np.linspace(0.01, 0.30, 60)
    dca_models = [
        ("eicu_setC",           "eICU Set C",                COL["navy"]),
        ("bidmc_postopB_firth", "BIDMC postop-B (deployed)", COL["rust"]),
    ]
    for ci, (key, label, color) in enumerate(dca_models):
        cp = CACHE / f"oof_{key}.npz"
        if not cp.exists():
            continue
        z = np.load(cp); y = z["y"].astype(int); p = z["p"].astype(float)
        nb_m, nb_all = net_benefit(y, p, thr)
        axB.plot(thr * 100, nb_m, lw=1.8, color=color, label=f"{label} — model")
        if ci == 0:
            axB.plot(thr * 100, nb_all, lw=1.0, color=COL["grey"], ls="--",
                      label="Treat all")
            axB.plot(thr * 100, np.zeros_like(thr), lw=1.0, color=COL["slate"],
                      ls=":", label="Treat none")
    axB.axvspan(5, 15, color=COL["soft"], alpha=0.20,
                 label="Clinical threshold band")
    axB.set_xlim(0, 30); axB.set_ylim(-0.04, 0.10)
    axB.set_xlabel("Probability threshold (%)")
    axB.set_ylabel("Net benefit")
    axB.set_title("Decision-curve net benefit")
    style_axis(axB, ygrid=True, xgrid=False)
    axB.axhline(0, color=COL["grey"], lw=0.5)
    add_panel_label(axB, "B")

    # Panel-local legend inside Panel B (rather than the shared bottom one).
    # Place in the upper-right corner where the curves are densest at the
    # left, leaving the upper-right empty.
    axB.legend(loc="upper right", fontsize=7.5, frameon=False,
                handlelength=2.0, handletextpad=0.5)

    plt.savefig(FIG / "F2_calibration_dca.png")
    plt.savefig(FIG / "F2_calibration_dca.pdf")
    plt.close()
    print("[OK] F2_calibration_dca — JNNP style")


# ─── F3 — Eleven-method battery ─────────────────────────────
def figure_3():
    imb = pd.read_csv(RES / "21_imbalance_sweep.csv")
    firth = pd.read_csv(RES / "24_firth_bayes_lr.csv")
    stack = pd.read_csv(RES / "22_diverse_stacking.csv")

    rows = []
    for _, r in imb[imb["feature_set"] == "postop_A"].iterrows():
        rows.append({"method": r["method"], "auc": r["auc"], "ci_lo": r["ci_lo"],
                      "ci_hi": r["ci_hi"], "brier": r["brier"]})
    for _, r in firth[firth["feature_set"] == "postop_A"].iterrows():
        if r["model"] not in ("BalancedRF (baseline)",):
            rows.append({"method": r["model"], "auc": r["auc"],
                          "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                          "brier": r["brier"]})
    for _, r in stack[stack["feature_set"] == "postop_A"].iterrows():
        if "baseline" not in r["model"].lower():
            rows.append({"method": r["model"], "auc": r["auc"],
                          "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                          "brier": r["brier"]})
    df = pd.DataFrame(rows).drop_duplicates(subset="method").reset_index(drop=True)
    df = df.sort_values("auc", ascending=True).reset_index(drop=True)

    is_firth   = df["method"].str.contains("Firth", case=False, na=False)
    is_baseline = df["method"].str.contains("baseline", case=False, na=False)
    colors = [COL["rust"] if f else COL["navy"] if b else COL["soft"]
              for f, b in zip(is_firth, is_baseline)]

    # Wider figure (9.5" instead of 7.5") so the long method names on the
    # AUC panel's y-axis no longer crowd the Brier bars.  width_ratios
    # tilted further toward the AUC panel because that is the panel with
    # the long labels.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 6.4),
                              gridspec_kw={"width_ratios": [1.9, 1.0]})
    plt.subplots_adjust(wspace=0.08, bottom=0.22, top=0.90,
                          left=0.20, right=0.97)
    fig.suptitle("BIDMC development cohort  ·  postoperative-A feature set",
                 fontsize=10.5, fontweight="bold", color=COL["navy"], y=0.975)

    # Panel A: AUC + CI
    axA = axes[0]
    pos = np.arange(len(df))
    axA.errorbar(df["auc"], pos,
                  xerr=[df["auc"] - df["ci_lo"], df["ci_hi"] - df["auc"]],
                  fmt="none", ecolor=COL["slate"], elinewidth=0.8, capsize=2)
    for i, c in enumerate(colors):
        axA.scatter(df["auc"].iloc[i], pos[i], color=c, s=42, zorder=3,
                     edgecolor="black", linewidth=0.5)
    axA.set_yticks(pos)
    axA.set_yticklabels(
        [m.replace(" (baseline)", "") for m in df["method"]],
        fontsize=7.5)
    axA.axvline(0.5, ls=":", color=COL["grey"], lw=0.7)
    axA.set_xlim(0.45, 0.80)
    axA.set_xlabel("Cross-validated AUC (95% CI)")
    axA.set_title("Discrimination")
    style_axis(axA, ygrid=False, xgrid=True)
    add_panel_label(axA, "A")

    # Panel B: Brier (calibration)
    axB = axes[1]
    axB.barh(pos, df["brier"], color=colors, edgecolor="black", linewidth=0.4)
    axB.set_yticks(pos); axB.set_yticklabels([])
    axB.axvline(0.073, ls="--", color=COL["grey"], lw=0.7)
    axB.set_xlabel("Brier score (lower = better)")
    axB.set_title("Calibration")
    style_axis(axB, ygrid=False, xgrid=True)
    add_panel_label(axB, "B")

    # Shared legend below both panels
    from matplotlib.lines import Line2D
    legend_handles = [
        mpatches.Patch(facecolor=COL["rust"], edgecolor="black", linewidth=0.4),
        mpatches.Patch(facecolor=COL["navy"], edgecolor="black", linewidth=0.4),
        mpatches.Patch(facecolor=COL["soft"], edgecolor="black", linewidth=0.4),
        Line2D([0],[0], color=COL["grey"], ls="--", lw=1.2),
        Line2D([0],[0], color=COL["grey"], ls=":", lw=1.2),
    ]
    legend_labels = [
        "Firth penalized LR (deployment)",
        "BalancedRandomForest",
        "Other sensitivity models",
        "Base-rate variance (0.073)",
        "Chance (AUC = 0.5)",
    ]
    figure_legend_below(fig, legend_handles, legend_labels, ncol=3,
                         y=0.02, fontsize=7.5)

    plt.savefig(FIG / "F3_method_battery.png")
    plt.savefig(FIG / "F3_method_battery.pdf")
    plt.close()
    print("[OK] F3_method_battery — JNNP style")


# ─── F4 — Conformal (rebuilt native, DEPLOYED Firth postop-B) ──────────
def figure_4():
    # Sourced from the DEPLOYED candidate model (Firth postop-B), not the
    # BalancedRF conformal base. See results/44_conformal_postopB_firth.csv.
    out = pd.read_csv(RES / "44_conformal_postopB_firth.csv").sort_values("alpha")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    plt.subplots_adjust(wspace=0.30, bottom=0.18, top=0.90)

    # Panel A: class-conditional coverage validation
    axA = axes[0]
    a_line = np.linspace(0.01, 0.30, 60)
    # Shade the region at/above the target line: points landing here mean the
    # coverage guarantee is kept.
    axA.fill_between(a_line, 1 - a_line, 1.0, color=COL["forest"], alpha=0.08,
                      lw=0, zorder=0)
    axA.text(0.295, 0.995, "guarantee met\n(on or above the line)",
              fontsize=6.6, color=COL["forest"], va="top", ha="right",
              linespacing=1.2, zorder=1)
    axA.plot(a_line, 1 - a_line, color=COL["grey"], ls="--", lw=0.8,
              label="Target (1−α)")
    axA.plot(out["alpha"], out["coverage_class1"], marker="o",
               color=COL["rust"], lw=1.6, ms=6,
               markeredgecolor="black", markeredgewidth=0.4,
               label="Seizure class (class 1)")
    axA.plot(out["alpha"], out["coverage_class0"], marker="s",
               color=COL["navy"], lw=1.6, ms=6,
               markeredgecolor="black", markeredgewidth=0.4,
               label="No-seizure class (class 0)")
    axA.set_xlim(0, 0.30); axA.set_ylim(0.72, 1.00)
    axA.set_xlabel("α (target miscoverage)")
    axA.set_ylabel("Empirical coverage")
    axA.set_title("Class-conditional coverage")
    style_axis(axA, ygrid=True, xgrid=True)
    add_panel_label(axA, "A")
    axA.legend(loc="lower left", fontsize=7.5, frameon=False,
                handlelength=2.0, handletextpad=0.5)

    # Panel B: clinical utility (confident decisions)
    axB = axes[1]
    axB.plot(out["alpha"], out["rule_out_rate"], marker="o",
               color=COL["navy"], lw=1.8, ms=6,
               markeredgecolor="black", markeredgewidth=0.4,
               label="Rule-out")
    axB.plot(out["alpha"], out["rule_in_rate"], marker="s",
               color=COL["rust"], lw=1.8, ms=6,
               markeredgecolor="black", markeredgewidth=0.4,
               label="Rule-in")
    # Working-point callout at α=0.10 (90% target coverage): mark the rule-out
    # and rule-in points and annotate the confident-vs-defer split so the
    # operating point is readable straight off the figure.
    sub = out[out["alpha"] == 0.10]
    if len(sub) > 0:
        ro = float(sub["rule_out_rate"].iloc[0])
        ri = float(sub["rule_in_rate"].iloc[0])
        conf = ro + ri
        axB.axvline(0.10, color=COL["grey"], ls=":", lw=0.7)
        axB.annotate(
            f"α = 0.10:\n{conf*100:.0f}% confident\n"
            f"({ro*100:.0f}% rule-out + {ri*100:.0f}% rule-in)\n"
            f"{(1-conf)*100:.0f}% defer",
            xy=(0.10, max(ro, ri)), xytext=(0.05, 0.37),
            fontsize=6.8, color=COL["slate"], va="top", linespacing=1.3,
            arrowprops=dict(arrowstyle="->", color=COL["slate"], lw=0.8))
    axB.set_xlim(0, 0.30); axB.set_ylim(0, 0.50)
    axB.set_xlabel("α (target miscoverage)")
    axB.set_ylabel("Fraction of patients")
    axB.set_title("Confident-decision yield")
    style_axis(axB, ygrid=True, xgrid=True)
    add_panel_label(axB, "B")
    axB.legend(loc="upper left", fontsize=7.5, frameon=False,
                handlelength=2.0, handletextpad=0.5)

    plt.savefig(FIG / "F4_conformal.png")
    plt.savefig(FIG / "F4_conformal.pdf")
    plt.close()
    print("[OK] F4_conformal — JNNP style (deployed Firth postop-B)")


# ─── F5 — CEA (decision tree + plane + CEAC) ────────────────
def figure_5():
    # Native rebuild from the deployable postop-B PSA (the decision tree is a
    # supplementary figure). Panel A: cost-effectiveness plane (incremental vs
    # observation); Panel B: cost-effectiveness acceptability curves.
    psa = pd.read_csv(RES / "38_postopB_psa.csv")
    labels = {"obs": "Observation", "aed": "Universal AED",
              "mla": "ML-guided AED", "mlg": "ML-guided cEEG"}
    cmap = {"obs": COL["grey"], "aed": COL["rust"],
            "mla": COL["ochre"], "mlg": COL["navy"]}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.6))
    plt.subplots_adjust(wspace=0.34, bottom=0.20, top=0.90)

    # Panel A — CE plane: each active strategy incremental to observation
    for s in ["aed", "mla", "mlg"]:
        dq = (psa[f"qaly_{s}"] - psa["qaly_obs"])
        dc = (psa[f"cost_{s}"] - psa["cost_obs"])
        axA.scatter(dq, dc, s=3, alpha=0.10, color=cmap[s], edgecolors="none")
        axA.scatter(dq.mean(), dc.mean(), s=70, color=cmap[s],
                    edgecolors="black", linewidths=0.5, zorder=5, label=labels[s])
    axA.axhline(0, color=COL["grey"], lw=0.6)
    axA.axvline(0, color=COL["grey"], lw=0.6)
    xl = axA.get_xlim()
    xs = np.array([min(0, xl[0]), max(0, xl[1])])
    axA.plot(xs, 100_000 * xs, ls="--", color=COL["ochre"], lw=0.8,
             label="WTP $100k/QALY")
    axA.set_xlabel("Incremental QALYs vs observation")
    axA.set_ylabel("Incremental cost vs observation (US$)")
    axA.set_title("Cost-effectiveness plane")
    style_axis(axA, ygrid=True, xgrid=True)
    add_panel_label(axA, "A")
    axA.legend(loc="lower right", fontsize=6.8, frameon=False)

    # Panel B — CEAC: probability each strategy is optimal vs WTP
    S = ["obs", "aed", "mla", "mlg"]
    wtp = np.arange(0, 200_001, 5_000)
    prob = {s: [] for s in S}
    for w in wtp:
        nmb = np.column_stack([w * psa[f"qaly_{s}"].values - psa[f"cost_{s}"].values
                               for s in S])
        win = np.argmax(nmb, axis=1)
        for i, s in enumerate(S):
            prob[s].append((win == i).mean())
    for s in S:
        axB.plot(wtp / 1000, prob[s], color=cmap[s], lw=1.8, label=labels[s])
    axB.axvline(100, color=COL["grey"], ls=":", lw=0.7)
    axB.set_xlim(0, 200); axB.set_ylim(0, 1)
    axB.set_xlabel("WTP threshold (US$1000/QALY)")
    axB.set_ylabel("Probability strategy is optimal")
    axB.set_title("Cost-effectiveness acceptability")
    style_axis(axB, ygrid=True, xgrid=True)
    add_panel_label(axB, "B")
    axB.legend(loc="center right", fontsize=6.8, frameon=False)

    plt.savefig(FIG / "F5_cea.png")
    plt.savefig(FIG / "F5_cea.pdf")
    plt.close()
    print("[OK] F5_cea — native JNNP (CE plane + CEAC)")


# ─── F6 — VOI (rebuilt native) ──────────────────────────────
def figure_6():
    # Decision-sensitivity / value-of-information under the deployable model.
    # Panel A: one-way net-benefit swings (deterministic). Panel B: two-way
    # optimal-strategy map over AED efficacy x AED disutility.
    tor = pd.read_csv(RES / "42_aed_tornado.csv").sort_values("swing")
    grid = pd.read_csv(RES / "39_aed_harm_threshold.csv")
    grid = grid[grid["scan"] == "grid"].copy()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 4.2),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})
    plt.subplots_adjust(wspace=0.5, left=0.30, right=0.97, bottom=0.16, top=0.90)

    # Panel A — one-way tornado: AED efficacy & harm dominate
    pos = np.arange(len(tor))
    top = tor["parameter"].isin(["AED efficacy (RRR)", "AED disutility"])
    colors = [COL["rust"] if t else COL["navy"] for t in top]
    axA.barh(pos, tor["swing"], color=colors, edgecolor="black", linewidth=0.4)
    axA.set_yticks(pos)
    axA.set_yticklabels(tor["parameter"], fontsize=7.5)
    axA.set_xlabel("Net-benefit swing, ML-guided vs universal AED (US$/patient)")
    axA.set_title("One-way sensitivity")
    style_axis(axA, ygrid=False, xgrid=True)
    add_panel_label(axA, "A")

    # Panel B — two-way optimal-strategy map
    rrrs = sorted(grid["aed_rrr"].unique(), reverse=True)      # x
    us = sorted(grid["param"].unique())                        # y (disutility)
    Z = np.zeros((len(us), len(rrrs)))
    for gi, u in enumerate(us):
        for gj, r in enumerate(rrrs):
            row = grid[(grid["param"] == u) & (grid["aed_rrr"] == r)]
            Z[gi, gj] = 1 if (len(row) and row["winner"].iloc[0] != "aed") else 0
    from matplotlib.colors import ListedColormap
    axB.imshow(Z, aspect="auto", origin="lower",
               cmap=ListedColormap([COL["rust"], COL["navy"]]), vmin=0, vmax=1)
    axB.set_xticks(range(len(rrrs))); axB.set_xticklabels([f"{r:.2f}" for r in rrrs], fontsize=7.5)
    axB.set_yticks(range(len(us))); axB.set_yticklabels([f"{u:.2f}" for u in us], fontsize=7.5)
    axB.set_xlabel("AED relative-risk reduction")
    axB.set_ylabel("AED disutility")
    axB.set_title("Optimal strategy")
    for sp in axB.spines.values():
        sp.set_visible(True)
    add_panel_label(axB, "B")

    legend_handles = [
        mpatches.Patch(facecolor=COL["navy"], edgecolor="black", linewidth=0.4),
        mpatches.Patch(facecolor=COL["rust"], edgecolor="black", linewidth=0.4),
    ]
    figure_legend_below(fig, legend_handles,
                        ["ML-guided allocation optimal", "Universal AED optimal"],
                        ncol=2, y=0.02, fontsize=7.5)

    plt.savefig(FIG / "F6_voi.png")
    plt.savefig(FIG / "F6_voi.pdf")
    plt.close()
    print("[OK] F6_voi — native JNNP (one-way tornado + two-way strategy map)")


def main():
    print(f"Building JNNP-style main figures (font: {JNNP_FONT})")
    print()
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    print("\nAll 6 main figures rebuilt in JNNP aesthetic.")
    print("Output: figures/F[1-6].{png,pdf}  300 dpi PNG · vector PDF")


if __name__ == "__main__":
    main()
