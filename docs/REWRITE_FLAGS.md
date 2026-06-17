# Rewrite Flags — number-integrity catches for the manuscript revision

Issues surfaced during the Phase-0–3 re-analysis that the manuscript text must be
corrected to match. Each is a place where a previously stated number does not survive an
honest recomputation. The adversarial review pass (Phase 4) must verify each is fixed.

## F1 — Calibration slope of the deployed postop-B model is NOT ~1.0
- **Claimed (old manuscript / S6 note):** post-Platt calibration slope ≈ 0.99–1.04.
- **Honest OOF recomputation (`scripts/40`, per-repeat, 5×5):** slope ≈ **2.8 (SD 1.0)**;
  averaged-cache variant ≈ 3.1. CITL ≈ −0.001, ECE ≈ 0.034, Brier ≈ 0.068, AUC ≈ 0.633.
- **Why:** the model under-disperses risk (predicted probabilities compressed into ~[0.06,
  0.13] because discrimination is weak); the slope is also imprecisely estimated over that
  narrow logit range with 48 events. A "slope ≈ 1" almost certainly came from the Platt
  scaler's in-sample fit (≈1 by construction), not the honest out-of-fold slope.
- **Fix in text:** report calibration evidence as CITL ≈ 0 + low ECE + low Brier (all robust),
  and state the slope honestly (>1, under-dispersion) as a limitation consistent with the weak
  discrimination — do NOT claim slope ≈ 1. Source: `results/40_postopB_calibration_post_platt.csv`.
- **Magnitude depends on the Platt protocol** (`scripts/43`): direct-train Platt → slope ≈ 1.6;
  nested-3-fold Platt → slope ≈ 2.8. Both > 1. Report the value for whichever protocol is
  described in Methods (deployment uses nested), and state the slope honestly either way.
- **C1 corollary (FLIC/FLAC):** raw Firth is over-dispersed (slope 0.55); FLIC corrects only the
  intercept (CITL→0, slope stays 0.55); Platt corrects both → best Brier/ECE. This *justifies*
  the Platt choice over Firth's own FLIC/FLAC correction (cite Puhr 2017).
  Source: `results/43_flic_vs_platt.csv`.

## F2 — CEA optimal strategy (see ADR 0005)
- **Old:** "ML-guided cEEG is optimal at $100k/QALY."
- **Honest:** conditional on AED efficacy/harm; ML-guided allocation is preferred across the
  cSDH-plausible range (RRR ≤ 0.30 or disutility ≥ 0.10); universal AED wins only under
  optimistic imported assumptions. All active strategies beat observation.
- **Fix:** M3 rewording per ADR 0005 / `results/38_PHASE0_CEA_NUMBERS_MEMO.md`.

## F3 — AED efficacy parameter (aed_rrr) was imported, not cSDH-grounded
- **Old base case:** RRR 0.45 (TBI/tumour literature).
- **Honest:** no cSDH study shows a significant protective effect (own meta-analysis OR 2.62,
  95% CI 0.53–13.06). Base case RRR ≈ 0; ≤0.30 as optimistic bound.
- **Fix:** S11 + Methods CEA parameter table; cite `docs/aed_efficacy_harm_lit_review.md`.

## F4 — "02_calibration" postop-B row is the BalancedRandomForest base, not Firth
- The `bidmc_postopB` row in `results/02_calibration_metrics.csv` (Brier 0.199, slope 0.34) is
  the BRF conformal base model, NOT the deployed Firth model. Any text citing "postop-B
  calibration" must point to the Firth numbers in `results/40_*`, not this row.

## F5 — LOHO "I²=0%" is correct on the logit scale but scale-sensitive (S7, strengthening not correction)
- **Reproduced:** the manuscript's full/Set_C pooled AUC 0.684 (0.651–0.714), I²=0% is exactly the
  **logit-scale** (delta-method) random-effects result (`scripts/41`, `04_loho`). Set_A logit I²=2.8%.
- **Sensitivity:** on the **raw-AUC scale** I²=33.7% (Set_A) / 55.7% (Set_C) with significant Q
  (p=0.019 / p<0.0001). The logit scale stabilises variance and down-weights extreme-AUC sites,
  collapsing I² toward 0 — defensible, but the homogeneity claim is not scale-robust.
- **Fix in text (S7):** keep logit-scale I²≈0% as primary, but ADD τ² (now reported under DL /
  Paule-Mandel / REML), the per-site descriptive table (`results/41_loho_per_site.csv`), and the
  **prediction interval** as the honest transportability summary (logit PI ~[0.65,0.72] for Set_C;
  raw-scale PI ~[0.53,0.95]). Do not present I²=0% as proof of homogeneity without the PI.

## C2 — Conformal coverage under domain shift (resolved by Discussion, per plan)
Per the revision plan, C2's preferred resolution for this revision is a Discussion paragraph
(full leave-one-hospital-out conformal-coverage analysis is a prospective-study item). Text to
add (see [ADR 0004](adr/0004-conformal-to-action-mapping.md)): conformal guarantees hold under
exchangeability; under site-level domain shift, deployment should recalibrate the conformal
quantiles on each site's first ~50 cases to restore class-conditional coverage. No new figure
required for this submission.

## F6 — Population EVPI ($190M) is stale; recompute under the new CEA parameterisation
- The "$190M EVPI at $100k/QALY over 10 years" predates the deployable-postop-B operating point
  and the literature-grounded AED parameters (RRR≈0, higher disutility). Decision uncertainty is
  now larger, so EVPI is likely higher, but the specific figure must be **recomputed** before
  quoting.
- **Done in the rewrite:** the abstract/discussion no longer quote $190M; they state VOI ranks
  AED efficacy and AED harm as priorities (qualitative, robust). Recompute the EVPI/EVPPI number
  (`16_voi_evpi.py` under postop-B Se/Sp + new AED priors) before re-inserting any dollar figure.

_Phase 3 analytical items complete (S6, S12, S7, S11, C1 done; C2 by Discussion)._
_Phase 4 in progress: Title/Abstract/Discussion/Methods/Results rewritten + appendices S6–S8._

---

# Adversarial-review findings (Phase 4 critique by 4 independent agents)

## F7 — CEA circularity / "observation is dominated" vs "AED RRR≈0" (SUBSTANTIVE, all agents)
Under RRR≈0 with AED harm, universal AED cannot *dominate* observation — yet the text says "all
active strategies dominated observation." ML-guided "winning" partly reflects treating fewer people
with a near-useless drug (its edge scales with specificity, not seizure signal).
**Fix:** (a) reconcile observation-domination with low RRR; (b) add a random-allocation-at-matched-
treated-fraction comparator to prove the *model* drives value; (c) present base case (RRR 0.45) →
universal AED wins, ML wins only in the sensitivity region — don't tilt the prose. (Needs analysis + decision.)

## F8 — AED RRR base case stated inconsistently (FIXED)
Methods (no-effect/0.30) vs Results (0.45); model default is 0.45. Fixed: Methods now anchors the
base case at 0.45 (optimistic reference) with cSDH-plausible 0–0.30 in sensitivity.

## F9 — Conformal numbers (37%/27%/11%) are from postop-A/BalancedRF, not deployment Firth/postop-B
Deployment-model conformal is 35%/25%/10%. Fix: recompute/report conformal on Firth postop-B or
state the conformal base model. (Needs analysis.)

## F10 — VOI headline contradicts its own EVPPI table
Abstract/Discussion say AED efficacy/harm dominate VOI, but Table S5 ranks cEEG cost #1, prevalence
#2; AED harm trivial. Also the VOI is the stale postop-A run (F6). Fix: recompute VOI under postop-B,
align headline with table. (Needs analysis.)

## F11 — "deployable/deployment" (~10×) vs intended "candidate" (polish)
"deployment model"→"candidate/lead model"; "deployable set"→"leakage-safe set"; "deployable
operating point"→"candidate operating point."

## F12 — eICU is a related (mixed-acuity ICU) population, not cSDH-evacuation
cSDH-faithful strata are 0.57–0.61 (own Tables S2/S7); 0.750 is the least-cSDH-like slice. Fix: call
it "external evaluation in a related population," demote 0.750, foreground the prediction interval.

## F13 — Minor concordance (polish)
Brier 0.069/0.068/0.067 → standardise to 0.068; Title "calibrated"→"calibration-focused"; abstract
should carry the prediction interval and slope>1 caveat; "biological ceiling"→"sample-size/measured-
feature limit"; degenerate Firth coeffs (demographic/procedures — drop or explain); 7→8 supplementary
figures; international cost_scale 0.79/0.92 (CSV) vs 0.800/0.900 (table); replace placeholder refs
14–16; real IRB number; cross-database phenotype "agreement" asserted not demonstrated — soften.
