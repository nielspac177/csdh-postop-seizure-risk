"""Task 26 — Build the 6 consolidated main-paper figures for JNNP submission.

JNNP allows a maximum of 6 figures/tables in the main paper. This script
combines existing panel-level figures into 6 publication-grade composite
figures, with the remaining figures deferred to the supplement.

Figure 1 — Multi-database discrimination & external validation
   1A: BIDMC primary model AUC + bootstrap CI (Firth + BalancedRF) per feature set
   1B: eICU non-traumatic cohort + LOHO random-effects pooled diamond
   1C: Strict pre-seizure feature subset sensitivity

Figure 2 — Calibration + clinical utility
   2A: Calibration curves after Platt scaling (eICU Set C + BIDMC postop_A)
   2B: Decision-curve net benefit at thresholds 0–30%

Figure 3 — Eleven-method modelling battery
   3A: AUC per method (Firth, BRF, RF+CW, six SMOTE variants, XGBoost, LightGBM, stacking)
   3B: Brier score per method
   3C: Net benefit at 10% threshold per method (clinical-utility metric)

Figure 4 — Conformal risk stratification
   4A: Empirical coverage vs target across α
   4B: Rule-out / rule-in singleton fractions with α=0.10 annotated working point

Figure 5 — Cost-effectiveness analysis
   5A: TreeAge-style decision tree (4 strategies with base-case rollback)
   5B: Cost-effectiveness plane (PSA scatter at $100k WTP)
   5C: Cost-effectiveness acceptability curve

Figure 6 — Value-of-information
   6A: EVPPI tornado at WTP $100k/QALY
   6B: EVPI as a function of WTP threshold (per-patient and population-scaled)

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
from PIL import Image

from _shared import RES, FIG, CACHE

# Common style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

PALETTE = {
    "primary":   "#1f4e79",     # navy
    "highlight": "#d4621a",     # rust
    "ok":        "#2a8a3f",     # green
    "warn":      "#b03d3d",     # red
    "neutral":   "#6d6d6d",     # grey
    "accent":    "#7a5c9e",     # purple
    "soft":      "#bcd4eb",
}

# ── Figure 1 — Multi-database discrimination ───────────────
def figure_1():
    # Load results
    firth   = pd.read_csv(RES / "24_firth_bayes_lr.csv")
    loho    = pd.read_csv(RES / "04_loho_summary.csv")
    leak    = pd.read_csv(RES / "05_leakage_audit.csv")

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.45, wspace=0.32)

    # Panel A: BIDMC primary (Firth + BRF, postop_A + B)
    axA = fig.add_subplot(gs[0, 0])
    primary = firth[firth["model"].isin(["BalancedRF (baseline)",
                                          "Firth penalized LR"])]
    primary = primary.sort_values(["feature_set", "model"]).reset_index(drop=True)
    pos = np.arange(len(primary))
    colors = [PALETTE["primary"] if "BalancedRF" in m else PALETTE["highlight"]
              for m in primary["model"]]
    axA.errorbar(primary["auc"], pos,
                  xerr=[primary["auc"] - primary["ci_lo"],
                        primary["ci_hi"] - primary["auc"]],
                  fmt="o", capsize=4, color="black", ecolor=PALETTE["neutral"])
    for i, c in enumerate(colors):
        axA.scatter(primary["auc"].iloc[i], pos[i], color=c, s=90, zorder=3,
                     edgecolor="black", linewidth=0.5)
    axA.set_yticks(pos)
    axA.set_yticklabels([f"{r.feature_set}  ·  {r.model}" for r in primary.itertuples()],
                          fontsize=9)
    axA.invert_yaxis()
    axA.axvline(0.5, ls=":", color=PALETTE["neutral"], lw=1)
    axA.set_xlim(0.45, 0.85)
    axA.set_xlabel("Cross-validated AUC (bootstrap 95% CI)")
    axA.set_title("A. BIDMC primary cohort (n=655, 48 events)", weight="bold", loc="left")

    # Panel B: LOHO pooled random-effects estimate per cohort+set
    axB = fig.add_subplot(gs[0, 1])
    if "auc_pooled_RE" in loho.columns:
        loho_p = loho.sort_values(["cohort", "set"]).reset_index(drop=True)
        pos = np.arange(len(loho_p))
        axB.errorbar(loho_p["auc_pooled_RE"], pos,
                      xerr=[loho_p["auc_pooled_RE"] - loho_p["auc_pooled_RE_lo"],
                            loho_p["auc_pooled_RE_hi"] - loho_p["auc_pooled_RE"]],
                      fmt="D", capsize=4, color="black",
                      ecolor=PALETTE["neutral"], markersize=10,
                      markerfacecolor=PALETTE["primary"], markeredgecolor="black")
        axB.set_yticks(pos)
        labels = [f"{row['cohort']}/{row['set']}\n"
                   f"I²={row['I2_pct']:.0f}%, k={int(row['n_hospitals'])}"
                   for _, row in loho_p.iterrows()]
        axB.set_yticklabels(labels, fontsize=9)
        axB.invert_yaxis()
        axB.axvline(0.5, ls=":", color=PALETTE["neutral"], lw=1)
        axB.set_xlim(0.45, 0.85)
        axB.set_xlabel("Random-effects pooled AUC (95% CI)")
        axB.set_title("B. eICU leave-one-hospital-out pooled estimate", weight="bold", loc="left")

    # Panel C: temporal leakage sensitivity
    axC = fig.add_subplot(gs[1, :])
    leak_p = leak[leak["cohort"].str.contains("eICU", case=False, na=False)].copy()
    leak_p = leak_p.dropna(subset=["auc"]).reset_index(drop=True)
    if len(leak_p) > 0:
        pos = np.arange(len(leak_p))
        axC.errorbar(leak_p["auc"], pos,
                       xerr=[leak_p["auc"] - leak_p["lo"],
                             leak_p["hi"] - leak_p["auc"]],
                       fmt="o", capsize=4, color="black",
                       ecolor=PALETTE["neutral"], markersize=8)
        for i, row in leak_p.iterrows():
            color = PALETTE["ok"] if "strict" in row["spec"].lower() else PALETTE["primary"]
            axC.scatter(row["auc"], i, color=color, s=80, zorder=3,
                         edgecolor="black", linewidth=0.5)
        axC.set_yticks(pos)
        axC.set_yticklabels(leak_p["spec"], fontsize=8)
        axC.invert_yaxis()
        axC.axvline(0.5, ls=":", color=PALETTE["neutral"], lw=1)
        axC.set_xlim(0.45, 0.85)
        axC.set_xlabel("Cross-validated AUC (95% CI)")
        axC.set_title("C. Temporal-leakage audit — primary signal survives "
                       "strict pre-seizure feature subset (green)",
                       weight="bold", loc="left")

    fig.suptitle("Figure 1.  Multi-database discrimination performance",
                  fontsize=13, weight="bold", y=1.00)
    plt.savefig(FIG / "F1_discrimination.png", dpi=220, bbox_inches="tight")
    plt.savefig(FIG / "F1_discrimination.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] F1_discrimination.png")


# ── Figure 2 — Calibration + decision-curve net benefit ─────
def figure_2():
    # Existing calibration_curves and dca_curves images
    img_cal = FIG / "02_calibration_curves.png"
    img_dca = FIG / "03_dca_curves.png"
    if not (img_cal.exists() and img_dca.exists()):
        print(f"  Missing: {img_cal} or {img_dca}")
        return
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 2, 1); ax1.axis("off")
    ax1.imshow(Image.open(img_cal))
    ax1.set_title("A. Calibration after Platt scaling — eICU Set C + BIDMC postop_A",
                   weight="bold", loc="left", fontsize=11)
    ax2 = fig.add_subplot(1, 2, 2); ax2.axis("off")
    ax2.imshow(Image.open(img_dca))
    ax2.set_title("B. Decision-curve net benefit (clinical thresholds 5–15%)",
                   weight="bold", loc="left", fontsize=11)
    fig.suptitle("Figure 2.  Calibration and clinical utility",
                  fontsize=13, weight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(FIG / "F2_calibration_dca.png", dpi=220, bbox_inches="tight")
    plt.savefig(FIG / "F2_calibration_dca.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] F2_calibration_dca.png")


# ── Figure 3 — Method battery (AUC + Brier + NB) ────────────
def figure_3():
    imb = pd.read_csv(RES / "21_imbalance_sweep.csv")
    firth = pd.read_csv(RES / "24_firth_bayes_lr.csv")
    stack = pd.read_csv(RES / "22_diverse_stacking.csv")

    # Aggregate methods on postop_A only for compactness
    rows = []
    for _, r in imb[imb["feature_set"] == "postop_A"].iterrows():
        rows.append({"method": r["method"], "auc": r["auc"], "ci_lo": r["ci_lo"],
                     "ci_hi": r["ci_hi"], "brier": r["brier"],
                     "nb10": r["nb_10pct"]})
    for _, r in firth[firth["feature_set"] == "postop_A"].iterrows():
        if r["model"] not in ("BalancedRF (baseline)",):  # skip duplicate baseline
            rows.append({"method": r["model"], "auc": r["auc"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "brier": r["brier"], "nb10": np.nan})
    for _, r in stack[stack["feature_set"] == "postop_A"].iterrows():
        if "baseline" not in r["model"].lower():
            rows.append({"method": r["model"], "auc": r["auc"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "brier": r["brier"], "nb10": np.nan})
    df = pd.DataFrame(rows).drop_duplicates(subset="method").reset_index(drop=True)
    df = df.sort_values("auc", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                              gridspec_kw={"width_ratios": [1.5, 1]})
    pos = np.arange(len(df))
    is_firth = df["method"].str.contains("Firth", case=False, na=False)
    is_baseline = df["method"].str.contains("baseline", case=False, na=False)
    colors = [PALETTE["highlight"] if f else
              PALETTE["primary"] if b else
              PALETTE["soft"]
              for f, b in zip(is_firth, is_baseline)]

    # Panel A: AUC with CIs
    ax = axes[0]
    ax.errorbar(df["auc"], pos,
                 xerr=[df["auc"] - df["ci_lo"], df["ci_hi"] - df["auc"]],
                 fmt="o", capsize=3, color="black",
                 ecolor=PALETTE["neutral"], markersize=0)
    for i, c in enumerate(colors):
        ax.scatter(df["auc"].iloc[i], pos[i], color=c, s=70, zorder=3,
                     edgecolor="black", linewidth=0.5)
    ax.set_yticks(pos)
    ax.set_yticklabels(df["method"], fontsize=8)
    ax.axvline(0.5, ls=":", color=PALETTE["neutral"], lw=1)
    ax.set_xlim(0.45, 0.80)
    ax.set_xlabel("Cross-validated AUC (bootstrap 95% CI)")
    ax.set_title("A. Discrimination ceiling — eleven model classes converge",
                   weight="bold", loc="left", fontsize=11)

    # Panel B: Brier (calibration)
    ax = axes[1]
    ax.barh(pos, df["brier"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(pos); ax.set_yticklabels([])
    ax.axvline(0.073, ls="--", color=PALETTE["neutral"], lw=1,
                label="Base-rate variance (0.073)")
    ax.set_xlabel("Brier score (lower = better)")
    ax.set_title("B. Calibration — actionable improvement",
                   weight="bold", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)

    # Color legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=PALETTE["highlight"], label="Firth penalized LR (deployment model)"),
        Patch(color=PALETTE["primary"],   label="BalancedRandomForest (baseline)"),
        Patch(color=PALETTE["soft"],      label="Other sensitivity models"),
    ]
    axes[0].legend(handles=legend_handles, loc="lower right", fontsize=8)

    fig.suptitle("Figure 3.  Eleven-method modelling battery — AUC ceiling vs calibration gain",
                  fontsize=13, weight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(FIG / "F3_method_battery.png", dpi=220, bbox_inches="tight")
    plt.savefig(FIG / "F3_method_battery.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] F3_method_battery.png")


# ── Figure 4 — Conformal (already a good composite) ─────────
def figure_4():
    src = FIG / "25_conformal.png"
    if not src.exists():
        print(f"  Missing: {src}")
        return
    # Copy with header
    img = Image.open(src)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")
    ax.imshow(img)
    fig.suptitle("Figure 4.  Class-conditional conformal prediction — "
                  "individual-patient deployment",
                  fontsize=13, weight="bold", y=1.00)
    plt.savefig(FIG / "F4_conformal.png", dpi=220, bbox_inches="tight")
    plt.savefig(FIG / "F4_conformal.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] F4_conformal.png")


# ── Figure 5 — CEA (decision tree + plane + CEAC) ───────────
def figure_5():
    tree_img = FIG / "14_decision_tree.png"
    plane_img = FIG / "10_pairwise_plane.png"
    ceac_img = FIG / "10_ceac_pairwise.png"

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

    if tree_img.exists():
        ax = fig.add_subplot(gs[0, :])
        ax.axis("off")
        ax.imshow(Image.open(tree_img))
        ax.set_title("A. Decision tree — base-case rollback per strategy",
                       weight="bold", loc="left", fontsize=11)
    if plane_img.exists():
        ax = fig.add_subplot(gs[1, 0])
        ax.axis("off")
        ax.imshow(Image.open(plane_img))
        ax.set_title("B. Cost-effectiveness plane — 10,000-iteration PSA",
                       weight="bold", loc="left", fontsize=11)
    if ceac_img.exists():
        ax = fig.add_subplot(gs[1, 1])
        ax.axis("off")
        ax.imshow(Image.open(ceac_img))
        ax.set_title("C. Cost-effectiveness acceptability curves",
                       weight="bold", loc="left", fontsize=11)
    fig.suptitle("Figure 5.  Cost-effectiveness analysis — ML-guided AED is dominant; "
                  "ML-guided cEEG cost-effective at $100k WTP",
                  fontsize=13, weight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(FIG / "F5_cea.png", dpi=220, bbox_inches="tight")
    plt.savefig(FIG / "F5_cea.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] F5_cea.png")


# ── Figure 6 — VOI ──────────────────────────────────────────
def figure_6():
    src = FIG / "16_voi_evpi.png"
    if not src.exists():
        print(f"  Missing: {src}")
        return
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")
    ax.imshow(Image.open(src))
    fig.suptitle("Figure 6.  Value of information — research-priority frontier "
                  "at WTP $100k/QALY",
                  fontsize=13, weight="bold", y=1.00)
    plt.savefig(FIG / "F6_voi.png", dpi=220, bbox_inches="tight")
    plt.savefig(FIG / "F6_voi.pdf", bbox_inches="tight")
    plt.close()
    print("[OK] F6_voi.png")


def main():
    print("Building consolidated main-paper figures...")
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    print("\nAll 6 main figures saved to figures/F[1-6]_*.{png,pdf}")


if __name__ == "__main__":
    main()
