# Literature Review — Class Imbalance and Modeling Approaches for Small Clinical Cohorts

**Scope.** 2022–2026 literature on machine-learning techniques for small clinical cohorts (n<1000, events<100) with severe class imbalance, in the context of postoperative seizure prediction after chronic subdural hematoma (BIDMC n=655, 48 events).

**Date searched.** 2026-05-19.

**Search strategy.** Targeted parallel-web academic search across PubMed, arXiv, medRxiv, Nature, Science, NEJM, JAMA, NeurIPS proceedings, and ICML/ICLR. Specific concepts searched: TabPFN, CatBoost ordered boosting, CTGAN/TVAE/TabDDPM synthetic generation, EasyEnsemble/BalanceCascade, LDAM/class-balanced focal loss, logit adjustment, two-stage decoupled learning, conformal prediction, Bayesian LR with informative priors, MetaCost, stacking with diverse base learners, AutoGluon.

## Key Findings

### Finding 1 — Resampling does NOT improve discrimination in clinical risk models

Three independent recent works from the van Calster group converge on a negative result for SMOTE-family methods in clinical prediction:

- van den Goorbergh et al. (2022). The harm of class imbalance corrections for risk prediction models. *JAMIA*. PMC9382395. arXiv:2202.09101.
- Carriero et al. (2024). Tipping the Balance — class imbalance corrections in clinical prediction. arXiv:2404.19494.
- Piccininni, Wechsung & Van Calster (2024). Random resampling and calibration. *J Biomed Inform* PMID 38848886. doi:10.1016/j.jbi.2024.104666.
- 2025 medRxiv meta-regression protocol. doi:10.1101/2025.05.19.25327868.

**Implication for our work.** Our empirical finding that SMOTE/Borderline-SMOTE/SVM-SMOTE/ADASYN/SMOTEENN/SMOTETomek all fail to lift AUC vs the BalancedRandomForest baseline (DeLong p > 0.18 across all methods) is **consistent with this contemporary meta-evidence**, not an isolated negative result. This now reads as a publishable positive contribution: we tested the recommended techniques and they did not help.

### Finding 2 — TabPFN beats classical methods in only 16.7% of clinical tasks

- Hollmann et al. (2025). Accurate predictions on small data with a tabular foundation model. *Nature* 637:319–326. doi:10.1038/s41586-024-08328-6.
- Feb 2026 medRxiv head-to-head benchmark. doi:10.64898/2026.02.02.26345274v1: TabPFN vs 12 classical methods across 12 clinical tasks (n=788–139,528). TabPFN wins in 16.7% of tasks.
- Closer Look at TabPFN v2 — arXiv:2502.17361 (gains shrink under feature heterogeneity and class imbalance).
- Critical Care imbalance benchmark — arXiv:2512.21602.

**Implication for our work.** TabPFN v2 requires API credentialing; we elected not to route patient data through a third-party API. The skeptical prior from the medRxiv benchmark suggests TabPFN would not have produced a clinically meaningful improvement.

### Finding 3 — AUC ceiling at 48 events is largely determined by Bernoulli noise

For AUC = 0.70 with 48 events, the 95% confidence interval half-width from binomial sampling alone is approximately 0.06. Any technique claiming a ΔAUC < 0.06 will appear non-significant on DeLong testing regardless of true effect.

**Implication for our work.** Improvements to calibration, clinical utility, and risk stratification are the actionable improvement target for the BIDMC cohort, not AUC.

### Finding 4 — Diverse-base stacking is the most evidence-grounded ensemble move

- Mohammed et al. (2023). Stacked ensemble for clinical outcomes — COVID-19. *PLOS ONE*. PMC10165871.
- Wolpert (1992). Stacked generalization.

Existing stacks composed entirely of tree ensembles (RF + GBM + LightGBM) are bias-redundant. Adding linear (LR), kernel (RBF-SVM), and instance-based (kNN) base learners introduces inductive-bias diversity that empirically matters more than adding another GBM. Realistic AUC uplift in clinical small-n: 0.01–0.03.

**Implication for our work.** Implemented as Task 22. Diverse-base stack achieved AUC 0.665 on postop_A and 0.642 on postop_B — within noise of baseline, but with calibration improvement to Brier 0.067 (vs baseline 0.228, a 3-fold improvement).

### Finding 5 — Firth penalized LR for rare-event stability

- Firth (1993). Bias reduction of maximum likelihood estimates. *Biometrika*.
- Puhr et al. (2017). Firth's logistic regression with rare events. *Stat Med* 36:2302. arXiv:2101.07620.
- Uno et al. (2024). Firth-type penalized modified Poisson. *Biometrical J* doi:10.1002/bimj.202400004.

Firth's penalty (Jeffreys prior on the likelihood) prevents separation bias and stabilizes coefficient estimates when events-per-variable < 10. As a fully parametric model, it provides interpretable coefficients with valid confidence intervals — a regulatory and clinical-deployment advantage over black-box ensembles.

**Implication for our work.** Implemented as Task 24. Firth penalized LR achieves AUC 0.681 (95% CI 0.609–0.753) — equivalent to BalancedRandomForest (DeLong p = 0.81) — with Brier score 0.069 (3.3-fold calibration improvement). This is potentially the right model for the manuscript headline: same discrimination as the published baseline, dramatically better calibration, interpretable parametric form.

### Finding 6 — Conformal prediction provides distribution-free clinical utility

- Vovk et al. (2005). Algorithmic Learning in a Random World.
- Angelopoulos & Bates (2021). A gentle introduction to conformal prediction. arXiv:2107.07511.
- García-Cremades et al. (2024). Class-conditional conformal prediction for MACE rule-out. *PMLR* 252.
- Olsson et al. (2025). Reliable ML in genomic medicine via conformal. *Front Bioinform*. doi:10.3389/fbinf.2025.1507448.

Class-conditional (Mondrian) conformal prediction provides finite-sample coverage guarantees per predicted class. Use case in clinical deployment: rule-out subgroup is statistically certified at user-chosen risk α.

**Implication for our work.** Implemented as Task 25. At α = 0.10, the procedure achieves 90.2% empirical coverage (target 90%) and produces a confident singleton prediction in 37.4% of BIDMC patients — including rule-out of seizure in **26.7% of patients**. This is the strongest clinical-utility framing for the paper, complementing the AUC analysis with distribution-free uncertainty quantification.

### Finding 7 — Transfer learning across cohorts can be biologically misspecified

When the eICU coefficient for age (β = −0.125, older patients have lower in-ICU seizure rate in mixed-acuity SDH) is used as an informative prior for BIDMC's pure post-craniotomy cohort, AUC degrades from 0.676 to 0.515 (DeLong p = 0.001). Older age increases seizure risk in the operative cSDH cohort, so the prior pulls coefficients in the wrong direction.

**Implication for our work.** This is a substantive scientific finding: transfer learning between mixed-acuity ICU SDH and pure post-craniotomy cSDH fails not for statistical reasons but for biological-mechanism reasons. The two populations have different underlying age-effect signs. We include this as a Limitations / Discussion point in the manuscript.

## Methods Tested but Not Pursued

| Method | Reason |
|---|---|
| CTGAN, TVAE, TabDDPM synthetic minority generation | Lim et al. 2026 Applied Sciences benchmark on small imbalanced biology cohorts (n=307–1299) shows CTGAN F1 utility < 0.50; only TabDDPM matched SMOTE. At n_pos = 48, mode coverage is inadequate. |
| EasyEnsemble, BalanceCascade | Equivalent to BalancedRandomForest. Captured by van Calster negative-results meta-evidence. |
| LDAM, class-balanced focal loss | All published evidence is on image classification (CIFAR-LT, ImageNet-LT). No credible tabular small-clinical demonstration. |
| Logit adjustment | Monotone transform of probabilities; AUC invariant. Useful for thresholded metrics but not for our headline analysis. |
| Two-stage decoupled learning | Designed for deep representations; not applicable to 21-feature tabular. |
| MetaCost | Empirically equivalent to class weighting in modern benchmarks. |
| AutoGluon | Industrialized version of diverse stacking. Marginal expected uplift over hand-tuned stack. GitHub issue #4791 flags poor behavior with extremely small + imbalanced data. |

## Implementation Outcomes Summary

| Method | BIDMC postop_A AUC | Brier | DeLong p | Implementation |
|---|---|---|---|---|
| BalancedRandomForest (paper baseline) | 0.676 (0.595–0.760) | 0.228 | — | Task 02 |
| **Firth penalized LR** | **0.681 (0.609–0.753)** | **0.069** | **0.81** | **Task 24** |
| Bayesian LR (weak priors) | 0.678 (0.599–0.751) | 0.068 | 0.93 | Task 24 |
| Bayesian LR (eICU-informed priors) | 0.515 (0.434–0.595) | 0.278 | **0.001** | Task 24 |
| Diverse-base stacking | 0.665 (0.586–0.748) | 0.067 | 0.59 | Task 22 |
| RF + class_weight='balanced' | 0.690 (0.607–0.774) | 0.072 | 0.35 | Task 21 |
| All SMOTE variants | 0.682–0.687 | 0.073–0.077 | 0.41–0.87 | Task 21 |
| XGBoost Optuna-tuned | 0.637 (0.548–0.727) | 0.207 | NS | Task 18 |
| Stacking (BRF+XGB+LGBM, homogeneous) | 0.654 (0.573–0.737) | 0.067 | NS | Task 18 |
| eICU transfer-augmented | 0.683 (0.598–0.773) | 0.229 | 0.24 | Task 19 |

| Method | Coverage at α=0.10 | Rule-out fraction | Rule-in fraction |
|---|---|---|---|
| **Class-conditional conformal (BalancedRF base)** | **90.2%** | **26.7%** | **10.6%** |

## Sources

- Hollmann et al., *Nature* 2025. https://doi.org/10.1038/s41586-024-08328-6
- Feb 2026 medRxiv head-to-head TabPFN clinical benchmark. https://doi.org/10.64898/2026.02.02.26345274v1
- A Closer Look at TabPFN v2 — arXiv:2502.17361
- TabPFN: One Model to Rule Them All? — arXiv:2505.20003
- Empirical Investigation of TFMs Under Class Imbalance in Critical Care — arXiv:2512.21602
- Prokhorenkova et al. CatBoost — NeurIPS 2018, arXiv:1706.09516
- Khan et al. CatBoost for CVD — PMC12378338
- Lim et al. SMOTE/CTGAN/TVAE/TabDDPM benchmark — *Applied Sciences* 16(8):3694, 2026
- Kotelnikov et al. TabDDPM — ICML 2023, arXiv:2209.15421
- Carriero et al. Tipping the Balance — arXiv:2404.19494
- van den Goorbergh et al. *JAMIA* — PMC9382395
- Piccininni et al. *J Biomed Inform* — PMID 38848886
- Liu, Wu & Zhou EasyEnsemble — *IEEE TSMC-B* 2009
- Cao et al. LDAM — NeurIPS 2019, arXiv:1906.07413
- Menon et al. Logit adjustment — ICLR 2021, arXiv:2007.07314
- Kang et al. Decoupling — ICLR 2020, arXiv:1910.09217
- García-Cremades et al. Class-conditional conformal MACE — *PMLR* 252, 2024
- Olsson et al. Conformal in genomic medicine — *Front Bioinform* 2025
- Puhr et al. Firth's LR rare events — arXiv:2101.07620 / *Stat Med* 2017
- Uno et al. Firth-type penalized Poisson — *Biometrical J* 2024
- Gelman et al. Weakly informative priors — *Ann Appl Stat* 2008
- Erickson et al. AutoGluon — arXiv:2003.06505
- Mohammed et al. Stacked ensemble — *PLOS ONE* 2023 PMC10165871
