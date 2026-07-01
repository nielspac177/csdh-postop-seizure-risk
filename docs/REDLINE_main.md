# Redline — `main_manuscript.docx` (pre-revision intent → revised text)

Reviewer-facing summary of the substantive wording changes made during the revision,
organised by section and keyed to the rewrite flags (F1–F19) in
`github_repo/docs/REWRITE_FLAGS.md`. Old text is the pre-revision intent (original
`JNNP_main_manuscript.docx` / earlier abstract draft); new text is the rebuilt
`main_manuscript.docx`. Strikethrough marks deletions, **bold** marks insertions.

---

## Title (F13)

- OLD: "A calibrated and ~~conformally-deployable~~ risk score for postoperative seizure
  after chronic subdural haematoma evacuation: a proof-of-concept multi-database study…"
- NEW: "Postoperative seizure after chronic subdural haematoma evacuation: a
  **calibration-focused, conformal-prediction** proof-of-concept with multi-database
  **evaluation** and value-of-information analysis."
- Rationale: drop "deployable" claim (model is a candidate, not deployable);
  "calibrated" → "calibration-focused"; "validation/study" framing softened to
  "evaluation."

## Abstract — Methods (F12)

- OLD: external set described as "5,376 SDH ICU stays across 139 hospitals … (external)."
- NEW: "3,297 non-traumatic subdural haematoma ICU stays across 42 hospitals … (**external
  evaluation in a related population**; 300 seizures)."
- Rationale: eICU is a mixed-acuity ICU population, not cSDH-evacuation; demote it from
  "external validation" to "external evaluation in a related population."

## Abstract — Results (F1, F9/F14, F12, F13, F16, F6/F10)

- OLD: "Firth … selected as the **deployment model** … discriminated at AUC **0.681**…
  with **3.3-fold** better calibration (Brier **0.069** vs 0.228). eICU external AUC was
  0.750 …, I²=0%. Conformal pre[diction] [37%/27% partition implied]."
- NEW: "Firth … was the **candidate model**, fit on a **leakage-safe postoperative-B**
  set; it discriminated at **AUC 0.645** (the AUC 0.681 postoperative-A set is **not
  deployable**, within the Bernoulli noise floor). … Brier **0.068** … calibration-in-the-
  large near zero, though predictions remained **under-dispersed (recalibration slope >1)**.
  … random-effects pooled AUC 0.684 (**prediction interval 0.53–0.95**). Class-conditional
  conformal prediction at α=0.10 gave confident singletons in **22%** (rule-out **11%**,
  rule-in **11%**). ML-guided allocation beat observation and, at a matched treated
  fraction, beat random allocation (**discrimination premium ≈$1,000–1,600/patient**).
  Population EVPI ≈ **$23M** over 10 years."
- Rationale: report the leakage-safe headline AUC (0.645), not the peri-decision 0.681;
  Brier standardised to 0.068 (F13); calibration reported honestly as CITL≈0 + slope>1
  rather than "slope≈1" (F1); add prediction interval (F5/F13); conformal corrected to
  the deployed-model 22% partition (F9/F14); add the model-vs-random discrimination
  premium sourced to new Table S6d (F16); replace stale $190M EVPI with recomputed $23M
  (F6/F10).

## Abstract — Conclusion (F2, F7, F18)

- OLD: "Small-cohort clinical machine learning can be honestly **deployable** … **ML-guided
  AED prophylaxis is cost-effective**; VOI prioritises per-day cEEG cost, baseline seizure
  prevalence and AED efficacy."
- NEW: "…yields a **candidate model that meets methodological preconditions for
  prospective clinical evaluation**. **Whether selective ML-guided prophylaxis is
  preferable to treating all or none hinges on AED efficacy and harm in cSDH, which
  current evidence leaves uncertain**; value-of-information prioritises resolving these."
- Rationale: remove the unconditional "deployable" + "cost-effective" claims; present the
  CEA decision as conditional on uncertain AED parameters (F2/F7).

## Methods — CEA / AED parameters (F3, F8)

- OLD: AED relative-risk reduction base case 0.45 (TBI/tumour literature) treated as the
  reference; Methods and Results inconsistent (0.30 vs 0.45).
- NEW: base case anchored at 0.45 as the **optimistic imported reference**, with the
  **cSDH-plausible range (RRR ≤ 0.30, no significant protective effect)** carried in
  sensitivity; Methods and Results now consistent. cSDH-grounded AED priors cited.

## Methods / Results / Table 1 legend — terminology (F11, F18)

- "deployment model" / "deployed model" → **candidate model**; "deployment pipeline" →
  **candidate pipeline**; "deployable set" → **leakage-safe set**; "deployable / external
  validation" of the eICU cohort → **external evaluation**. "candidate operating point",
  "leakage-safe", filenames and TRIPOD-checklist references unchanged.

## Results — Conformal (F9, F14)

- OLD: confident singletons for ~37%/27% of patients (BalancedRF / postop-A base).
- NEW: "On the **candidate Firth postoperative-B model**, class-conditional coverage
  90.3% (no-seizure) / 93.8% (seizure) at α=0.10; **confident singleton in 22%** of
  patients — **rule-out 11%, rule-in 11%**, with **78% deferred** to clinical judgment."
- Figure 4 regenerated on the deployed Firth model (`44_conformal_postopB_firth.csv`);
  caption and image now agree (rule-out 11%, rule-in 11%, defer 78%).

## Results / Discussion — CEA framing (F2, F7, F19)

- OLD: "all active strategies dominated observation; ML-guided AED prophylaxis is
  cost-effective / optimal at $100k/QALY."
- NEW: choice between universal AED and ML-guided allocation is **conditional on AED
  efficacy and harm**; under cSDH-plausible values ML-guided wins, universal AED only
  under optimistic imported assumptions. Adds the **model-vs-random comparator**
  (discrimination premium ≈$1,000–1,600/patient, Table S6d) to show the **model**, not
  the treated fraction, creates value; acknowledges that near RRR≈0 the edge over
  observation derives substantially from the early-detection (cEEG) limb (F19).

## Results / Discussion — calibration slope (F1)

- OLD: post-Platt calibration slope ≈ 0.99–1.04 (≈1).
- NEW: CITL ≈ 0 with low ECE/Brier reported as the robust calibration evidence; **slope
  >1 (under-dispersion)** stated honestly as a limitation consistent with weak
  discrimination over a narrow probability range at 48 events.

## Discussion — value of information (F6, F10)

- OLD: "$190M EVPI at $100k/QALY over 10 years"; VOI headline (AED efficacy/harm) at odds
  with the EVPPI table (which ranked cEEG cost #1).
- NEW: VOI **recomputed under the postop-B candidate model and cSDH-grounded AED priors**
  (Table S5 = `45_voi_postopB.csv`): population **EVPI ≈ $23M** over 10 years; the
  narrative now states the model's discrimination carries **modest decision value**
  relative to AED efficacy and per-day cEEG cost — consistent with the table.

## Discussion — model limitation (F17, new)

- NEW (Appendix S8): the only statistically significant coefficients are **procedure
  variables (surgical decompression, MMA embolization, drainage)** reflecting **confounding
  by indication rather than causal seizure biology** (drainage OR 1.44 runs against a
  plausible causal protective effect) — a limitation of the candidate model.

## Discussion — heterogeneity / "biological ceiling" (F5, F13)

- OLD: "biological ceiling"; I²=0% presented as proof of homogeneity.
- NEW: "**sample-size / measured-feature limit**"; logit-scale I²=0% kept as primary but
  paired with **τ² estimators and the 95% prediction interval (0.53–0.95)** as the honest
  transportability summary; raw-AUC sensitivity I² up to ~56% disclosed.
