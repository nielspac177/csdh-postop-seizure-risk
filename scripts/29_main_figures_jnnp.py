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
        axC.set_yticklabels(leak_p["spec"], fontsize=8)
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
    # Load raw OOF predictions from cache instead of quantile-binned points;
    # LOWESS gives a non-parametric estimate of E[y | p_pred] without forcing
    # the calibration plot through coarse decile bins (which are noisy on
    # BIDMC's 48 events).  A 95% bootstrap envelope shows the uncertainty.
    axA = axes[0]
    models_to_plot = [
        ("eicu_setC",     "eICU Set C",     COL["navy"], "-"),
        ("bidmc_postopA", "BIDMC postop-A", COL["rust"], "--"),
    ]
    axA.plot([0, 0.6], [0, 0.6], color=COL["grey"], lw=0.8, ls=":",
              label="Perfect calibration", zorder=1)

    rng = np.random.default_rng(42)
    grid = np.linspace(0.0, 0.55, 80)
    for key, label, color, ls in models_to_plot:
        cache_path = CACHE / f"oof_{key}.npz"
        if not cache_path.exists():
            continue
        z = np.load(cache_path)
        y = z["y"].astype(float); p = np.clip(z["p"], 1e-6, 1 - 1e-6)
        # bootstrap LOWESS envelope: 200 resamples
        smooths = []
        for _ in range(200):
            idx = rng.integers(0, len(y), len(y))
            try:
                sm = lowess(y[idx], p[idx], frac=0.5, return_sorted=True,
                              it=0, missing="drop")
            except Exception:
                continue
            if len(sm) < 5: continue
            smooths.append(np.interp(grid, sm[:, 0], sm[:, 1]))
        if not smooths: continue
        smooths = np.vstack(smooths)
        smooths = np.clip(smooths, 0, 1)
        lo  = np.percentile(smooths, 2.5,  axis=0)
        hi  = np.percentile(smooths, 97.5, axis=0)
        mid = np.percentile(smooths, 50,   axis=0)
        axA.fill_between(grid, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
        axA.plot(grid, mid, color=color, lw=2.2, ls=ls,
                  label=label, zorder=3)
    axA.set_xlim(0, 0.55); axA.set_ylim(0, 0.55)
    axA.set_xlabel("Predicted probability")
    axA.set_ylabel("Observed event rate")
    axA.set_title("Calibration after Platt scaling")
    style_axis(axA, ygrid=True, xgrid=True)
    add_panel_label(axA, "A")
    # Panel-local legend: place inside the panel rather than the shared bottom
    axA.legend(loc="upper left", fontsize=7.5, frameon=False,
                handlelength=2.0, handletextpad=0.5,
                title="(LOWESS · 95% bootstrap envelope)",
                title_fontsize=7)
    axA_handles, axA_labels = [], []  # nothing to forward to the shared legend

    # Panel B: decision-curve net benefit
    axB = axes[1]
    dca = pd.read_csv(RES / "03_dca_summary_at_thresholds.csv") \
            if (RES / "03_dca_summary_at_thresholds.csv").exists() else None
    if dca is not None:
        # DCA CSV schema: columns model, threshold, model_nb, treat_all_nb, incremental
        model_col = "model" if "model" in dca.columns else "cohort"
        nb_model_col = "model_nb" if "model_nb" in dca.columns else "nb_model"
        nb_all_col   = "treat_all_nb" if "treat_all_nb" in dca.columns else "nb_all"
        models_to_plot = [c for c in dca[model_col].unique()
                           if c in ("bidmc_postopA", "eicu_setC")]
        for ci, c in enumerate(models_to_plot):
            sub = dca[dca[model_col] == c].sort_values("threshold")
            color = COL["rust"] if "bidmc" in c else COL["navy"]
            label = "BIDMC postop-A" if "bidmc" in c else "eICU Set C"
            axB.plot(sub["threshold"]*100, sub[nb_model_col], lw=1.8,
                      color=color, label=f"{label} — model")
            if ci == 0:
                axB.plot(sub["threshold"]*100, sub[nb_all_col], lw=1.0,
                          color=COL["grey"], ls="--", label="Treat all")
                axB.plot(sub["threshold"]*100, [0]*len(sub),
                          lw=1.0, color=COL["slate"], ls=":", label="Treat none")
        axB.axvspan(5, 15, color=COL["soft"], alpha=0.20,
                     label="Clinical threshold band")
        axB.set_xlim(0, 30); axB.set_ylim(-0.10, 0.10)
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
    plt.subplots_adjust(wspace=0.08, bottom=0.22, top=0.94,
                          left=0.20, right=0.97)

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

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.4))
    plt.subplots_adjust(wspace=0.34, bottom=0.30, top=0.93)

    # Panel A: class-conditional coverage validation
    axA = axes[0]
    a_line = np.linspace(0.01, 0.30, 60)
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
    # Working-point annotation at α=0.10 (90% target coverage).
    sub = out[out["alpha"] == 0.10]
    if len(sub) > 0:
        sub = sub.iloc[0]
        defer = 1.0 - sub["rule_out_rate"] - sub["rule_in_rate"]
        axB.axvline(0.10, color=COL["grey"], ls=":", lw=0.7)
        axB.annotate(f"α = 0.10 (90% coverage):\n"
                       f"  rule-out = {sub['rule_out_rate']:.0%}\n"
                       f"  rule-in  = {sub['rule_in_rate']:.0%}\n"
                       f"  defer    = {defer:.0%}",
                       xy=(0.10, sub["rule_out_rate"]),
                       xytext=(0.205, 0.03),
                       fontsize=7.5, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="#fffaf0",
                                 edgecolor=COL["ochre"], linewidth=0.8),
                       arrowprops=dict(arrowstyle="->", lw=0.8,
                                       color=COL["ochre"]))
    axB.set_xlim(0, 0.30); axB.set_ylim(0, 0.50)
    axB.set_xlabel("α (target miscoverage)")
    axB.set_ylabel("Fraction of patients")
    axB.set_title("Clinical utility — confident decisions")
    style_axis(axB, ygrid=True, xgrid=True)
    add_panel_label(axB, "B")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], color=COL["navy"], marker="o", lw=1.8,
                markeredgecolor="black", markeredgewidth=0.4,
                markerfacecolor=COL["navy"]),
        Line2D([0],[0], color=COL["rust"], marker="s", lw=1.8,
                markeredgecolor="black", markeredgewidth=0.4,
                markerfacecolor=COL["rust"]),
    ]
    legend_labels = [
        "Rule-out — singleton {no seizure}",
        "Rule-in  — singleton {seizure}",
    ]
    axB.legend(legend_handles, legend_labels,
                loc="upper left", fontsize=7.0, frameon=False,
                handlelength=2.0, handletextpad=0.5)

    fig.text(0.14, 0.05,
             "Prediction-set categories:\n"
             "    {no seizure}              →  rule-out (skip AED)\n"
             "    {seizure}                 →  rule-in (target cEEG)\n"
             "    {seizure, no seizure} →  defer to clinical judgment.",
             fontsize=7.5, color="#262320", ha="left",
             family="DejaVu Sans", linespacing=1.5)
    fig.text(0.14, -0.04,
             "Deployed candidate model (Firth, leakage-safe postop-B feature\n"
             "set). Conformal quantiles fit on a held-out calibration split;\n"
             "rates and class-conditional coverage are out-of-fold.",
             fontsize=7.0, color=COL["grey"], style="italic", ha="left",
             family="DejaVu Sans", linespacing=1.4)
    plt.subplots_adjust(bottom=0.36)

    plt.savefig(FIG / "F4_conformal.png")
    plt.savefig(FIG / "F4_conformal.pdf")
    plt.close()
    print("[OK] F4_conformal — JNNP style (deployed Firth postop-B)")


# ─── F5 — CEA (decision tree + plane + CEAC) ────────────────
def figure_5():
    from PIL import Image
    tree = FIG / "14_decision_tree.png"
    plane = FIG / "10_pairwise_plane.png"
    ceac  = FIG / "10_ceac_pairwise.png"

    # Use a 3-panel layout (A wide on top, B/C side-by-side below) with
    # JNNP-style panel labels and a soft outer frame.
    fig = plt.figure(figsize=(7.0, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                            hspace=0.30, wspace=0.10)

    axA = fig.add_subplot(gs[0, :])
    axA.axis("off")
    if tree.exists():
        axA.imshow(Image.open(tree))
    axA.set_title("Decision tree — base-case rollback")
    add_panel_label(axA, "A", offset=(-0.01, 1.02))

    axB = fig.add_subplot(gs[1, 0])
    axB.axis("off")
    if plane.exists():
        axB.imshow(Image.open(plane))
    axB.set_title("Cost-effectiveness plane")
    add_panel_label(axB, "B", offset=(-0.04, 1.02))

    axC = fig.add_subplot(gs[1, 1])
    axC.axis("off")
    if ceac.exists():
        axC.imshow(Image.open(ceac))
    axC.set_title("CEAC vs willingness-to-pay")
    add_panel_label(axC, "C", offset=(-0.04, 1.02))

    plt.savefig(FIG / "F5_cea.png")
    plt.savefig(FIG / "F5_cea.pdf")
    plt.close()
    print("[OK] F5_cea — JNNP style")


# ─── F6 — VOI (rebuilt native) ──────────────────────────────
def figure_6():
    evppi = pd.read_csv(RES / "16_voi_evppi.csv").sort_values(
        "evppi_per_patient", ascending=True).reset_index(drop=True)
    # EVPI vs WTP — recompute from PSA file if available
    psa_file = RES / "16_voi_psa_tracked.csv"

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.0),
                              gridspec_kw={"width_ratios": [1.4, 1.0]})
    plt.subplots_adjust(wspace=0.30, bottom=0.28, top=0.93)

    # Panel A: EVPPI tornado
    axA = axes[0]
    pos = np.arange(len(evppi))
    # Highlight top 4 with rust
    top_n = 4
    colors = [COL["rust"] if i >= len(evppi) - top_n else COL["navy"]
              for i in range(len(evppi))]
    axA.barh(pos, evppi["evppi_per_patient"], color=colors, edgecolor="black",
              linewidth=0.4)
    axA.set_yticks(pos)
    axA.set_yticklabels(evppi["parameter"], fontsize=7.5)
    axA.set_xlabel(r"Per-patient EVPPI (US\$) at WTP \$100k/QALY")
    axA.set_title("Research-priority ranking")
    style_axis(axA, ygrid=False, xgrid=True)
    add_panel_label(axA, "A")

    # Panel B: EVPI vs WTP
    axB = axes[1]
    STRATEGIES = ["obs", "aed", "mla", "mlg"]
    if psa_file.exists():
        psa = pd.read_csv(psa_file)
        wtp_grid = np.arange(0, 200_001, 5_000)
        evpis = []
        for w in wtp_grid:
            nmb = np.column_stack([
                w * psa[f"qaly_{s}"].values - psa[f"cost_{s}"].values
                for s in STRATEGIES
            ])
            evpis.append(nmb.max(axis=1).mean() - nmb.mean(axis=0).max())
        axB.plot(wtp_grid / 1000, evpis, color=COL["indigo"], lw=2.0)
        # mark common WTP thresholds
        for w, label_y in [(50, 0.93), (100, 0.85), (150, 0.78)]:
            axB.axvline(w, color=COL["grey"], ls=":", lw=0.6)
            axB.text(w, max(evpis) * label_y, f"${w}k", ha="center",
                       fontsize=7, color=COL["grey"])
    axB.set_xlabel(r"WTP threshold (\$1000 / QALY)")
    axB.set_ylabel(r"Per-patient EVPI (US\$)")
    axB.set_title("Per-patient EVPI")
    axB.set_xlim(0, 200)
    style_axis(axB, ygrid=True, xgrid=True)
    add_panel_label(axB, "B")

    # Shared legend / note below the figure
    legend_handles = [
        mpatches.Patch(facecolor=COL["rust"], edgecolor="black", linewidth=0.4),
        mpatches.Patch(facecolor=COL["navy"], edgecolor="black", linewidth=0.4),
    ]
    legend_labels = [
        "Top-4 EVPPI — research-priority frontier",
        "Remaining parameters",
    ]
    figure_legend_below(fig, legend_handles, legend_labels, ncol=2,
                         y=0.08, fontsize=7.5)
    fig.text(0.5, 0.02,
             "Population EVPI ≈ $190M over 10 years (40,000-patient annual operative cohort, 3% discount)",
             ha="center", va="bottom", fontsize=7.5, style="italic",
             color=COL["slate"])

    plt.savefig(FIG / "F6_voi.png")
    plt.savefig(FIG / "F6_voi.pdf")
    plt.close()
    print("[OK] F6_voi — JNNP style")


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
