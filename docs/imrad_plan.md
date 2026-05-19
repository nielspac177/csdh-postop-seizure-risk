# Manuscript plan — manuscript submission

## Target journal
**Journal of Neurology, Neurosurgery and Psychiatry** (the target journal). Author guidelines (2026): Original Research max 4,000 words main text; structured abstract 250 words; max 6 figures/tables combined in main paper; Vancouver references typically 30–50; TRIPOD-AI reporting for prediction models; supplementary material allowed.

---

## Five main messages — sharpened after critical-thinking + brainstorming

**Master frame:** the paper is a *proof of concept that small-cohort clinical machine learning can be honestly deployable when calibration and decision-integration replace AUC as the optimization target.*

1. **The discrimination ceiling at AUC ≈ 0.68 is biologically real, not a modelling failure.** With 48 events the Bernoulli noise floor on AUC = 0.70 has 95% CI half-width ≈ 0.06. We confirm the ceiling with an eleven-method battery — six SMOTE-family oversamplers, Optuna-tuned XGBoost and LightGBM, a diverse-base stacking ensemble, two Bayesian parameterizations, and transfer-learning augmentation from eICU — and align it with three independent 2022–2025 meta-analyses (van den Goorbergh JAMIA 2022; Carriero arXiv 2024; Piccininni J Biomed Inform 2024). A hostile reviewer cannot ask "did you try method X" because we did.

2. **Calibration replaces AUC as the actionable target for clinically deployable small-cohort risk scores.** Firth penalized logistic regression matches BalancedRandomForest discrimination (AUC 0.681 vs 0.676; DeLong p = 0.81) with a 3.3-fold improvement in Brier score (0.069 vs 0.228). The Firth model is parametric, interpretable, supplies valid coefficient confidence intervals, and aligns with TRIPOD-AI reporting requirements. Decision-curve net benefit at the clinically-anchored 5–15% probability band confirms the calibration gain translates into clinical utility.

3. **Conformal prediction translates the model into individual-patient decisions with distribution-free coverage guarantees.** Class-conditional (Mondrian) conformal prediction at α = 0.10 produces a confident singleton prediction in 37% of patients — rule-out of seizure in 27% (AED-avoidance candidates) and rule-in in 11% (intensive-monitoring candidates), with empirical coverage of 90.2%. The remaining 63% of patients are *explicitly deferred to clinical judgment*; the model does not overclaim. This is the first application of distribution-free conformal sets to postoperative-seizure decision support.

4. **Decision-analytic integration shows ML guidance dominates current practice, with first-in-field value-of-information prioritization.** Under literature-refreshed inputs (Ceribell-era cEEG cost, US WTP $100k/QALY, geriatric AED adverse-event burden), ML-guided AED prophylaxis dominates observation and universal AED on both cost and QALY axes. Probabilistic sensitivity over 10,000 Monte Carlo iterations captures the AUC-driven uncertainty; the accompanying value-of-information analysis is the first such application to postoperative-seizure prophylaxis and quantifies population-level research value at $190M over a 10-year horizon, ranking per-day cEEG cost ($195/pt EVPPI), seizure prevalence ($127), and AED relative-risk reduction ($96) as the priority research targets.

5. **Two methodological tools released for the field.** First, a corrected ICD-10 outcome definition for nationwide population-scale analyses of postoperative seizure after cSDH that separates acute symptomatic seizure (R56.x, 780.39, G41.x) from pre-existing epilepsy (G40.x, 345.x); the previously-reported population signal (AUC 0.617) collapses to chance (0.498) under the corrected definition. Second, a documented case of biological transfer-learning failure between mixed-acuity ICU SDH and pure post-craniotomy cSDH cohorts: eICU's negative age coefficient pulls BIDMC predictions in the wrong direction (AUC drops from 0.676 to 0.515), demonstrating that cohort biology, not statistics, drives the failure. Both are released as deployable code on GitHub for replication.

---

## Critical-thinking review — what hostile reviewers might say and how each bullet handles it

| Reviewer concern | Defence |
|---|---|
| "You didn't try the latest SOTA tabular model" | Bullet 1 cites 11-method battery + 2022–2025 meta-evidence + Bernoulli noise floor. Ceiling does not depend on our experiments alone. |
| "Calibration is just a different metric" | Bullet 2 binds calibration to TRIPOD-AI reporting + decision-curve net benefit at clinically anchored thresholds. |
| "What about the 63% you can't classify with conformal?" | Bullet 3 reframes as appropriate deferral to clinical judgment; the model does not overclaim. |
| "AUC 0.68 → garbage-in, garbage-out for CEA" | Bullet 4 binds to probabilistic sensitivity + VOI explicitly quantifies the uncertainty as a research-prioritization tool. |
| "You're nitpicking other people's work" | Bullet 5 reframes corrections as released tools (cleaned ICD-10 codeset; transfer-learning failure documented as a generalizable warning). |

---

## Brainstorm — alternative framings considered

- *"Beat the SOTA"* — rejected: AUC ceiling is real, framing as discrimination winner would invite obvious refutation.
- *"Deep-learning frontier"* — rejected: deep methods (TabPFN, NN architectures) have negative benchmarks at n_pos < 50 (Feb 2026 medRxiv 16.7% win rate). Would require infrastructure we cannot defend.
- *"Calibration-first methodology"* — adopted as bullet 2's frame.
- *"Decision-integration template for small-cohort neurosurgery prediction"* — adopted as the master framing.
- *"Honest reckoning / negative-result-done-right"* — adopted as bullet 1's frame.
- *"Transfer-learning failure case study"* — adopted as bullet 5's second tool.

---

## IMRAD skeleton (the target journal word budget 4,000 main text)

### Introduction (~600 words)
- Para 1: cSDH as one of the most common neurosurgical diseases of ageing populations; rising incidence; mortality and disability burden.
- Para 2: Postoperative seizure complicates 7–12% of cSDH operations; impact on ICU stay, mortality, function.
- Para 3: Current decision controversy — universal AED prophylaxis (harm in elderly: falls, cognition) vs selective monitoring (cEEG cost, capacity). Identifying high-risk patients requires accurate, calibrated, decision-integrated risk stratification.
- Para 4: Limits of prior ML attempts in cSDH seizure — small samples, no calibration, no clinical translation, no external validation, often outcome misclassification.
- Para 5: Our aim — proof of concept. We develop and externally validate a deployment-ready risk score, embed it in conformal prediction sets for individual-patient decisions, and integrate it in a refreshed decision-analytic model with value-of-information analysis.

### Methods (~1,200 words)
- 2.1 Study design and ethics. Multi-database retrospective; IRB/PHI compliance per database; TRIPOD-AI reporting checklist (supplement).
- 2.2 Cohort assembly.
  - BIDMC (development): all consecutive cSDH surgical evacuations 2010–2023.
  - eICU CRD v2.0 (external): SDH ICU stays 2014–2015; primary stratum "non-traumatic" + sensitivity to four cohort definitions.
  - NIS 2016–2019 (population): ICD-10 chronic + surgical SDH; outcome reclassified to acute symptomatic seizure only.
  - MIMIC-IV (NLP pipeline only).
- 2.3 Outcome. Postoperative seizure (R56.x, 780.39, G41.x; EMR clinical-seizure indicator). For NIS, both original "combined" definition (conflated with G40.x/345.x) and corrected "acute symptomatic only" definition reported.
- 2.4 Features. Postoperative-A (21 features, includes AED-timing/EEG) and postoperative-B (18 features, leakage-safe).
- 2.5 Models (11 classes). Baseline BalancedRandomForest; deployment Firth penalized LR; sensitivity: class-weighted RF, six SMOTE variants, Optuna-tuned XGBoost/LightGBM, diverse-base stacking ensemble (LR+RF+XGB+KNN+RBF-SVM), Bayesian LR with weak / eICU-informed priors.
- 2.6 Robustness battery. Temporal-leakage audit, cohort-definition sensitivity, leave-one-hospital-out with DerSimonian–Laird random-effects pooling (Hanley–McNeil variance), bootstrap 95% CIs on AUC and calibration metrics, decision-curve net benefit, competing-risks Cox + IPCW Fine–Gray (Geskus), λ-path-tuned group-LASSO.
- 2.7 Conformal prediction. Class-conditional (Mondrian) conformal with split-conformal scheme, 75/25 calibration split, α ∈ {0.05, 0.10, 0.20}.
- 2.8 NLP pipeline. Regex-pattern radiology report extractor; manually-labelled validation set.
- 2.9 Cost-effectiveness analysis. 4-strategy decision tree → 10-year Markov; literature-refined inputs; 10,000-iteration probabilistic sensitivity; EVPI / EVPPI via Strong–Oakley non-parametric regression on 5,000 PSA samples.
- 2.10 Reproducibility. n_jobs = 1 throughout; SEED = 42; code released on GitHub.

### Results (~1,400 words)
- 3.1 Cohort characteristics (Table 1).
- 3.2 Primary discrimination — Firth LR + BalancedRF on BIDMC; eICU external; LOHO forest (Figure 1).
- 3.3 Calibration after Platt scaling; decision-curve net benefit at 5–15% thresholds (Figure 2).
- 3.4 Eleven-model comparison sweep — none lift AUC, calibration improves universally (Figure 3 + supplementary Table S1).
- 3.5 Conformal risk stratification — coverage validation + rule-out/rule-in (Figure 4).
- 3.6 NIS outcome reclassification — methodological correction (supplementary).
- 3.7 Cost-effectiveness analysis + decision tree (Figure 5).
- 3.8 Value-of-information — population EVPI + EVPPI tornado (Figure 6).

### Discussion (~800 words)
- 4.1 Principal findings — proof-of-concept: discrimination ceiling acknowledged; calibration, conformal, CEA, VOI as the substantive contributions.
- 4.2 Comparison with prior literature — alignment with 2022–2025 meta-evidence on class imbalance.
- 4.3 Methodological strengths — multi-database, robustness battery, conformal deployment, EVPI prioritization.
- 4.4 Limitations — small-n ceiling, single-institution development, biology mismatch in transfer, administrative outcomes, US-payer perspective.
- 4.5 Clinical implications — actionable rule-out for 27% of patients, decision-analytic dominance of ML-AED, research-priority ranking.
- 4.6 Future directions — imaging-NLP augmentation, prospective conformal-set validation, international cEEG cost data collection.

---

## Figure & table allocation

### Main paper (6 maximum per the target journal guidelines)
1. **Figure 1** — Multi-database discrimination forest plot (BIDMC Firth, eICU Set C, LOHO pooled).
2. **Figure 2** — Calibration curves (after Platt scaling) + decision-curve net benefit.
3. **Figure 3** — Eleven-model AUC + Brier comparison (combines current Figures 7 + 8).
4. **Figure 4** — Conformal prediction: coverage + rule-out/rule-in.
5. **Figure 5** — Decision tree with base-case rollback + cost-effectiveness plane.
6. **Figure 6** — Value-of-information: EVPPI tornado + EVPI vs WTP.

### Table 1 (main)
- Cohort characteristics across BIDMC / eICU / NIS.

### Supplementary
- Table S1: Full 11-model class comparison with all metrics.
- Table S2: Cohort-definition sensitivity (eICU strata).
- Table S3: Calibration metrics with bootstrap 95% CIs (all 6 cohort-model combinations).
- Table S4: CEA base-case + PSA summary.
- Table S5: EVPPI ranking.
- Figure S1: Study flow diagram.
- Figure S2: LOHO per-hospital forest plot (all 42 hospitals).
- Figure S3: Time-window leakage audit (eICU 0–24h, 0–48h, 0–72h, ≥24h).
- Figure S4: Missingness diagnostics + Rubin's-rules MI pooling.
- Figure S5: NIS outcome reclassification — original vs corrected.
- Figure S6: Competing-risks survival diagnostics + Schoenfeld residuals.
- Figure S7: Radiology NLP pipeline validation.
- Appendix S1: TRIPOD-AI reporting checklist.
- Appendix S2: Reproducibility code links + GitHub commit hash.
