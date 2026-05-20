# What conformal prediction adds to a clinical ML model

Targeted literature review (2019-2026) supporting the new methodology slide
in the 15-minute talk. Each value-add is paired with the strongest single
recent reference.

## Six unique contributions of conformal prediction

1. **Distribution-free finite-sample coverage.** Conformal prediction guarantees
   P(true label ∈ prediction set) ≥ 1−α with no asymptotic, distributional,
   or model-correctness assumptions — only data exchangeability. Calibration
   (Platt, isotonic) makes no coverage claim; Bayesian credible intervals
   need correct prior + likelihood specifications.
   - Angelopoulos AN, Bates S. *A Gentle Introduction to Conformal Prediction
     and Distribution-Free Uncertainty Quantification.* Foundations & Trends
     in ML, 2023. arXiv:2107.07511.

2. **Model-agnostic post-hoc wrapper.** Conformal is applied after fitting;
   the Firth coefficients and the inferential machinery they preserve are
   untouched. Bayesian alternatives need re-specifying and re-fitting; Platt
   scaling modifies the probability head.
   - Vovk V, Gammerman A, Shafer G. *Conformal prediction: A unified review
     of theory and new challenges.* Bernoulli 29(1):1-23, 2023. arXiv:2005.07972.

3. **Adaptive per-instance set size.** Conformal returns {}, {no-seizure},
   {seizure}, or {no-seizure, seizure}. Set size grows for ambiguous
   patients and shrinks for easy ones, directly encoding heterogeneous
   epistemic uncertainty that a calibrated probability collapses.
   - Romano Y, Sesia M, Candès EJ. *Classification with Valid and Adaptive
     Coverage* (APS). NeurIPS, 2020. arXiv:2006.02544.

4. **Class-conditional (Mondrian) coverage for imbalanced clinical data.**
   Marginal coverage can be satisfied by always covering the majority class.
   Mondrian / label-conditional CP calibrates within each class, so the
   seizure-positive coverage rate is guaranteed separately — something
   neither Platt, isotonic, nor naive bootstrap CIs deliver on imbalanced
   data.
   - Löfström T, Boström H, Linusson H, Johansson U. *Mondrian Cross-Conformal
     Prediction on Imbalanced Bioactivity Data.* J Chem Inf Model 57(7):
     1591-1598, 2017. doi:10.1021/acs.jcim.7b00159.
   - Theoretical basis: Vovk V. *Conditional Validity of Inductive Conformal
     Predictors.* PMLR 25, 2012.

5. **Computational cost orders of magnitude below MCMC.** Split-conformal
   needs one held-out quantile computation. Bayesian uncertainty for a
   hierarchical Firth-type model needs MCMC and prior tuning. Conformal
   delivers comparable coverage in seconds and adds no hyperparameters or
   priors — meaningful for re-deployment and external validation.
   - Fortuna L et al. *Conformal prediction for uncertainty quantification
     in dynamic biological systems.* PLoS Comput Biol, 2025.
     doi:10.1371/journal.pcbi.1013098.

6. **A principled "I don't know" for clinical deployment.** A doubleton set
   {seizure, no-seizure} is a formal abstention signal that triggers EEG
   monitoring or specialist review. A null set {} flags out-of-distribution
   patients. Fixed thresholds force a binary call; calibrated probabilities
   leave the clinician to invent a cut-off.
   - Olsson H et al. *Conformal selective prediction with cost-aware deferral
     for safe clinical triage under distribution shift.* Scientific Reports,
     2026. doi:10.1038/s41598-026-40637-w.

## Comparison vs the alternatives we considered and discarded

| Property | Platt/isotonic calibration | Bayesian credible intervals | Bootstrap CIs on predictions | Conformal (Mondrian) |
|---|:-:|:-:|:-:|:-:|
| Finite-sample coverage guarantee | × | × (asymptotic) | × (asymptotic) | **✓** |
| Distribution-free | × | × | × | **✓** |
| Model-agnostic post-hoc wrapper | partial | × | partial | **✓** |
| Adaptive per-patient set size | × | partial | × | **✓** |
| Class-conditional on imbalanced data | × | × | × | **✓** |
| No priors / no MCMC needed | ✓ | × | ✓ | **✓** |
| Formal abstention signal | × | × | × | **✓** |
| Computational cost | low | high | medium | **low** |

## Bottom line for the slide

Calibration fixes probabilities but promises nothing about errors. Bayesian
intervals need correct priors and MCMC. Bootstrap CIs are asymptotic and
ignore class imbalance. Mondrian conformal prediction on top of Firth
penalized LR gives a finite-sample, class-conditional, deployment-ready
uncertainty layer at near-zero computational cost.
