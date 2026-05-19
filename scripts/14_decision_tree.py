"""Task 14 — TreeAge-style decision tree (Figure 4 for CEA).

Renders the 4-strategy decision tree with:
  □ decision node (root)
  ○ chance nodes (seizure y/n, detected y/n)
  ▷ terminal nodes (cost / QALY pair)

Uses the deterministic point-estimate parameters from 10_11_cea_pairwise.py and
performs expected-value rollback at each chance node so the tree shows both the
structure and the rolled-up E[Cost], E[QALY] per strategy.

Outputs:
  figures/14_decision_tree.{png,pdf,svg}
  results/14_decision_tree_rollback.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from _shared import RES, FIG

# ── Base-case parameters (point estimates from Params in 10_11_cea_pairwise.py)
# Kept here as plain dict for readability in the figure.
P = dict(
    p_seizure_base   = 0.12,
    aed_rrr          = 0.45,
    model_sens       = 0.842,
    model_spec       = 0.504,
    p_detect_ceeg    = 0.95,
    p_detect_clin    = 0.30,
    cost_ml          = 50.0,
    cost_aed_total   = 200.0 + 1507.0 + 441.0,
    cost_aed_total_ceeg = 200.0 + 1293.0 + 140.0,
    cost_ceeg_total  = 1500.0 * 3,
    cost_sz_detected = 2500.0 + (20.3 - 9.8) * 3500.0 + 1000.0,
    cost_sz_undet_mult = 1.3,
    qaly_baseline    = 0.75,
    qaly_sz_loss     = 0.10,
    qaly_aed_loss    = 0.02 * (66 / 365),
    qaly_aed_loss_ceeg = 0.02 * (21 / 365),
    horizon_qaly_no_sz = 7.5,   # ~10y * 0.75 discounted
    horizon_qaly_sz    = 6.8,
    horizon_qaly_undet = 6.3,
)

# ── Strategy logic — point-estimate rollback ───────────────────────────
def rollback_strategy(name):
    """Return list of leaves: (path_label, prob, cost, qaly)."""
    p_sz = P["p_seizure_base"]
    leaves = []

    if name == "obs":
        # No prophylaxis, clinical detection only
        p_det = P["p_detect_clin"]
        leaves.append((f"sz→det",  p_sz * p_det,
                       P["cost_sz_detected"],
                       P["horizon_qaly_sz"] - P["qaly_sz_loss"]))
        leaves.append((f"sz→miss", p_sz * (1 - p_det),
                       P["cost_sz_detected"] * P["cost_sz_undet_mult"],
                       P["horizon_qaly_undet"] - P["qaly_sz_loss"]))
        leaves.append((f"no sz",    1 - p_sz, 0.0,
                       P["horizon_qaly_no_sz"]))

    elif name == "aed_all":
        # Universal AED, clinical detection
        p_sz_treated = p_sz * (1 - P["aed_rrr"])
        p_det = P["p_detect_clin"]
        leaves.append(("AED→sz→det",  p_sz_treated * p_det,
                       P["cost_aed_total"] + P["cost_sz_detected"],
                       P["horizon_qaly_sz"] - P["qaly_sz_loss"] - P["qaly_aed_loss"]))
        leaves.append(("AED→sz→miss", p_sz_treated * (1 - p_det),
                       P["cost_aed_total"] + P["cost_sz_detected"] * P["cost_sz_undet_mult"],
                       P["horizon_qaly_undet"] - P["qaly_sz_loss"] - P["qaly_aed_loss"]))
        leaves.append(("AED→no sz",   1 - p_sz_treated,
                       P["cost_aed_total"],
                       P["horizon_qaly_no_sz"] - P["qaly_aed_loss"]))

    elif name == "ml_aed":
        # ML → AED if predicted high risk; clinical detection
        sens = P["model_sens"]; spec = P["model_spec"]
        tp = sens * p_sz; fn = (1 - sens) * p_sz
        fp = (1 - spec) * (1 - p_sz); tn = spec * (1 - p_sz)
        rrr = P["aed_rrr"]
        # TP: AED reduces seizure risk by rrr; if seizes, detected clinically
        leaves.append(("ML+→TP→prevent", tp * rrr,
                       P["cost_ml"] + P["cost_aed_total"],
                       P["horizon_qaly_no_sz"] - P["qaly_aed_loss"]))
        leaves.append(("ML+→TP→sz",     tp * (1 - rrr),
                       P["cost_ml"] + P["cost_aed_total"] + P["cost_sz_detected"],
                       P["horizon_qaly_sz"] - P["qaly_sz_loss"] - P["qaly_aed_loss"]))
        # FP: AED but never would have seized
        leaves.append(("ML+→FP",        fp,
                       P["cost_ml"] + P["cost_aed_total"],
                       P["horizon_qaly_no_sz"] - P["qaly_aed_loss"]))
        # FN: no AED, seizure occurs, clinical detection
        p_det = P["p_detect_clin"]
        leaves.append(("ML−→FN→det",   fn * p_det,
                       P["cost_ml"] + P["cost_sz_detected"],
                       P["horizon_qaly_sz"] - P["qaly_sz_loss"]))
        leaves.append(("ML−→FN→miss",  fn * (1 - p_det),
                       P["cost_ml"] + P["cost_sz_detected"] * P["cost_sz_undet_mult"],
                       P["horizon_qaly_undet"] - P["qaly_sz_loss"]))
        # TN: no AED, no seizure
        leaves.append(("ML−→TN",        tn,
                       P["cost_ml"],
                       P["horizon_qaly_no_sz"]))

    elif name == "ml_ceeg":
        # ML → cEEG monitoring if predicted high risk
        sens = P["model_sens"]; spec = P["model_spec"]
        tp = sens * p_sz; fn = (1 - sens) * p_sz
        fp = (1 - spec) * (1 - p_sz); tn = spec * (1 - p_sz)
        p_det = P["p_detect_ceeg"]   # cEEG much higher detection
        # High-risk branch: cEEG + targeted AED (lower-burden)
        leaves.append(("ML+→TP→det",   tp * p_det,
                       P["cost_ml"] + P["cost_ceeg_total"] + P["cost_aed_total_ceeg"]
                          + P["cost_sz_detected"],
                       P["horizon_qaly_sz"] - P["qaly_sz_loss"]
                          - P["qaly_aed_loss_ceeg"]))
        leaves.append(("ML+→TP→miss",  tp * (1 - p_det),
                       P["cost_ml"] + P["cost_ceeg_total"] + P["cost_aed_total_ceeg"]
                          + P["cost_sz_detected"] * P["cost_sz_undet_mult"],
                       P["horizon_qaly_undet"] - P["qaly_sz_loss"]
                          - P["qaly_aed_loss_ceeg"]))
        leaves.append(("ML+→FP",       fp,
                       P["cost_ml"] + P["cost_ceeg_total"],
                       P["horizon_qaly_no_sz"]))
        # Low-risk branch: no cEEG, clinical detection only
        p_det_lo = P["p_detect_clin"]
        leaves.append(("ML−→FN→det",  fn * p_det_lo,
                       P["cost_ml"] + P["cost_sz_detected"],
                       P["horizon_qaly_sz"] - P["qaly_sz_loss"]))
        leaves.append(("ML−→FN→miss", fn * (1 - p_det_lo),
                       P["cost_ml"] + P["cost_sz_detected"] * P["cost_sz_undet_mult"],
                       P["horizon_qaly_undet"] - P["qaly_sz_loss"]))
        leaves.append(("ML−→TN",       tn,
                       P["cost_ml"],
                       P["horizon_qaly_no_sz"]))
    return leaves

STRATEGIES = [
    ("obs",     "Observation\n(no AED, no cEEG)"),
    ("aed_all", "Universal AED\nprophylaxis"),
    ("ml_aed",  "ML-guided AED\n(targeted prophylaxis)"),
    ("ml_ceeg", "ML-guided cEEG\n+ targeted AED"),
]


def compute_ev(leaves):
    e_cost = sum(p * c for _, p, c, _ in leaves)
    e_qaly = sum(p * q for _, p, _, q in leaves)
    return e_cost, e_qaly


# ── TreeAge-style figure rendering ─────────────────────────────────────
NODE_DECISION = dict(shape="square", color="#1f77b4")
NODE_CHANCE   = dict(shape="circle", color="#2ca02c")
NODE_TERMINAL = dict(shape="triangle", color="#d62728")

def draw_node(ax, x, y, kind, size=0.012):
    if kind == "decision":
        p = mpatches.FancyBboxPatch((x - size, y - size), 2*size, 2*size,
                                     boxstyle="square,pad=0", linewidth=1.2,
                                     facecolor=NODE_DECISION["color"], edgecolor="black")
    elif kind == "chance":
        p = mpatches.Circle((x, y), size, facecolor=NODE_CHANCE["color"],
                            edgecolor="black", linewidth=1.0)
    else:
        p = mpatches.RegularPolygon((x, y), 3, radius=size * 1.1,
                                     orientation=-3.14159/2,
                                     facecolor=NODE_TERMINAL["color"], edgecolor="black",
                                     linewidth=0.8)
    ax.add_patch(p)


def render_tree():
    # Compute leaf layout first to size the figure properly
    all_leaves = [(key, name, rollback_strategy(key)) for key, name in STRATEGIES]
    total_leaves = sum(len(L) for _, _, L in all_leaves)
    # vertical spacing per leaf, plus gap between strategies
    leaf_v = 0.40
    strat_gap = 1.6
    fig_h = max(11, total_leaves * leaf_v + len(STRATEGIES) * strat_gap + 2)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.set_xlim(0, 16); ax.set_ylim(0, fig_h); ax.axis("off")
    ax.set_title("Figure 4 — Decision tree, cSDH postoperative seizure prevention\n"
                 "Base-case rollback per strategy   "
                 "□ decision  ○ chance  ▷ terminal",
                 fontsize=13, pad=14, weight="bold")

    # x-coordinates of the columns
    X_ROOT  = 0.6
    X_STRAT = 2.6
    X_LEAF  = 11.0
    X_EV    = 14.5

    # Root decision node centered vertically
    root_y = fig_h / 2
    draw_node(ax, X_ROOT, root_y, "decision", size=0.18)
    ax.text(X_ROOT, root_y + 0.6, "Choose\nstrategy",
            ha="center", va="bottom", fontsize=9, weight="bold")

    # Walk strategies from top down
    cursor_y = fig_h - 1.2     # top margin
    rollback_rows = []

    for key, name, leaves in all_leaves:
        n = len(leaves)
        block_h = n * leaf_v
        strat_y = cursor_y - block_h / 2

        # Branch from root to strategy chance node
        ax.plot([X_ROOT + 0.18, X_STRAT - 0.18], [root_y, strat_y],
                color="black", lw=0.9)
        # Strategy label box placed near the chance node
        line1, *rest = name.split("\n")
        line2 = " ".join(rest) if rest else ""
        label_text = line1 + ("\n" + line2 if line2 else "")
        ax.text(X_STRAT - 0.35, strat_y, label_text,
                ha="right", va="center",
                fontsize=9.5, weight="bold", color="#1f2d3d",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef3fa",
                          edgecolor="#5b7da7", linewidth=1.0))

        # Strategy chance node
        draw_node(ax, X_STRAT, strat_y, "chance", size=0.16)

        # Leaves stacked vertically
        e_cost, e_qaly = compute_ev(leaves)
        rollback_rows.append({"strategy": name.replace("\n", " "),
                              "E_cost_USD": round(e_cost, 0),
                              "E_QALY": round(e_qaly, 4),
                              "n_leaves": n})

        # vertical positions of leaves within this strategy's block
        if n == 1:
            ys = [strat_y]
        else:
            ys = [strat_y + block_h / 2 - leaf_v * (i + 0.5) for i in range(n)]

        for (label, prob, cost, qaly), ly in zip(leaves, ys):
            ax.plot([X_STRAT + 0.16, X_LEAF - 0.16], [strat_y, ly],
                    color="#333", lw=0.7)
            # Single consolidated branch label, placed above the branch
            # at a horizontal-ish portion near the leaf end. Each leaf has its
            # own vertical slot so labels never overlap.
            label_x = X_STRAT + (X_LEAF - X_STRAT) * 0.55
            label_y = ly + 0.06  # just above the leaf's y-coordinate
            ax.text(label_x, label_y,
                    f"{label}   p={prob:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                    style="italic", color="#333",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor="#bbb", linewidth=0.5))
            # terminal node
            draw_node(ax, X_LEAF, ly, "terminal", size=0.13)
            # cost / QALY readout right of the terminal
            ax.text(X_LEAF + 0.3, ly,
                    f"${cost:>7,.0f}    {qaly:.3f} QALY",
                    ha="left", va="center", fontsize=8, family="monospace",
                    color="#1a1a1a")

        # Rolled-up expected value
        ax.text(X_EV, strat_y, f"E[Cost] = ${e_cost:,.0f}\nE[QALY] = {e_qaly:.3f}",
                ha="left", va="center", fontsize=9.5, weight="bold",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#fff7e0", edgecolor="#aa7700", linewidth=1.2))

        cursor_y -= block_h + strat_gap

    # Legend
    handles = [
        mpatches.Patch(facecolor=NODE_DECISION["color"], edgecolor="black",
                       label="□ Decision node"),
        mpatches.Patch(facecolor=NODE_CHANCE["color"], edgecolor="black",
                       label="○ Chance node"),
        mpatches.Patch(facecolor=NODE_TERMINAL["color"], edgecolor="black",
                       label="▷ Terminal node"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.95,
              bbox_to_anchor=(0.01, 0.005))

    plt.savefig(FIG / "14_decision_tree.png", dpi=220, bbox_inches="tight")
    plt.savefig(FIG / "14_decision_tree.pdf", bbox_inches="tight")
    plt.savefig(FIG / "14_decision_tree.svg", bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(rollback_rows)
    df["delta_cost_vs_obs"] = df["E_cost_USD"] - df.loc[0, "E_cost_USD"]
    df["delta_QALY_vs_obs"] = df["E_QALY"] - df.loc[0, "E_QALY"]
    df["ICER_vs_obs"] = df.apply(
        lambda r: r["delta_cost_vs_obs"] / r["delta_QALY_vs_obs"]
        if r["delta_QALY_vs_obs"] > 0 else float("nan"),
        axis=1,
    )
    df.to_csv(RES / "14_decision_tree_rollback.csv", index=False)
    print(df.round(3).to_string(index=False))
    print("\n[OK] figures/14_decision_tree.{png,pdf,svg}  results/14_decision_tree_rollback.csv")


if __name__ == "__main__":
    render_tree()
