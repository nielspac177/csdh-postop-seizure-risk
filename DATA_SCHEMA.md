# Data schema (reviewer-access bundle)

This file documents the column-level schema of each file in the reviewer-access bundle. The schema is **the same data structure already produced by the public scripts**, so a reviewer can verify by inspection that no additional patient-level information is included.

## `bidmc_postopA_dev.csv` · 655 × 22

| Column | Type | Description | Source field |
|---|---|---|---|
| `id` | int | Random per-patient ID; not linkable to MRN | derived |
| `sex` | int (0/1) | Sex (1 = male) | EMR |
| `age` | int | Age at index admission, in years | EMR |
| `blood_type` | int (0–8) | ABO + Rh, ordinal encoded | EMR |
| `sdh_type` | int (0–3) | Chronic / acute / mixed / unknown | imaging report |
| `sdh_thickness` | float | Maximum hematoma thickness, mm | imaging report |
| `csdh_size_change` | int (−1/0/1) | Pre→post size change (smaller / equal / larger) | imaging report |
| `mid_shift` | float | Midline shift, mm | imaging report |
| `hematoma_lat` | int (1–3) | Left / right / bilateral | imaging report |
| `collection_density` | int (0–3) | Hypo / iso / hyper / mixed | imaging report |
| `preop_gcs` | int (3–15) | Pre-operative Glasgow Coma Scale total | EMR |
| `epilepsy_hx` | int (0/1) | Prior history of epilepsy | EMR |
| `num_prev_sdh` | int | Number of previous SDH episodes | EMR |
| `demographic` | int (0–4) | Demographic stratum encoding | derived |
| `procedures` | int (0–4) | Procedure-class encoding | derived |
| `surg_decompression` | int (0/1) | Decompressive craniectomy performed | EMR |
| `mma_embo` | int (0/1) | Middle-meningeal artery embolisation | EMR |
| `drainage` | int (0/1) | Drainage at the time of evacuation | EMR |
| `postop_gcs` | int (3–15) | OR-exit Glasgow Coma Scale total | EMR |
| `aed_timing_recoded` | int (0/1/2) | AED administered preop / postop / not given | EMR |
| `prop_aed` | int (0/1) | Prophylactic AED course | EMR |
| `ab_eeg` | int (0/1) | Abnormal EEG documented during admission | EMR |
| `seizure` | int (0/1) | Outcome — any postoperative seizure during index admission | EMR |

## `bidmc_postopB_dev.csv` · 655 × 19

Identical to `postopA` minus the three leakage-suspect columns: `aed_timing_recoded`, `prop_aed`, `ab_eeg`. This is the postoperative-B feature set evaluated in the manuscript as a temporal-leakage robustness check.

## `eicu_setC_full.csv` · 5,376 × 104

eICU non-traumatic and traumatic SDH ICU stays. Variables include:

- **Demographics and admission**: `age`, `sex`, `apacheadmissiondx`, `hospitalid`, `unittype`.
- **Acuity**: `apache_score`, `apache_gcs`, `gcs_admission`, `prior_seizures`, `pupil_abnormality`.
- **Comorbidity flags (binary)**: `hypertension`, `diabetes`, `afib`, `heart_failure`, `ckd`, `coagulopathy`, `alcohol`, `liver_disease`, `dementia`, `obesity`, `tobacco`, `prior_stroke`, `cad`.
- **Anticoagulant / antiplatelet flags**: `any_anticoagulant`, `any_antiplatelet`, `any_steroid`.
- **Procedure flags**: `craniotomy`, `burr_hole`, `mechanical_ventilation`, `icp_monitor`.
- **First-24-h lab values**: `Hgb_first`, `PT___INR_first`, `WBC_x_1000_first`, `albumin_first`, `calcium_first`, `creatinine_first`, `glucose_first`, `lactate_first`, `magnesium_first`, `platelets_x_1000_first`, `potassium_first`, `sodium_first`, `total_bilirubin_first`.
- **24-h mean vitals**: `temperature_mean`, `sao2_mean`, `heartrate_mean`, `respiration_mean`, `systemicsystolic_mean`, `systemicdiastolic_mean`, `systemicmean_mean`.
- **24-h derived GCS**: `gcs_24h`, `gcs_min_24h`, `gcs_max_24h`, `gcs_delta`, `n_gcs_assessments`.
- **48-h trajectories (slope, delta, coefficient of variation) for**: sodium, potassium, glucose, creatinine, platelets, lactate, WBC, Hgb, heart rate, systemic systolic, systemic mean, temperature, SaO₂.
- **Pre-admission AED and prophylactic AED**: `pre_admission_aed`, `prophylactic_aed`.
- **Outcome**: `seizure` (binary, in-ICU postoperative seizure), `seizure_offset_min` (minutes from admission to seizure if any), `status_epilepticus`.

A full machine-readable schema CSV (`eicu_setC_schema.csv`) accompanies the data bundle on release.

## `eicu_setC_pure.csv` · 3,255 × 104

Same columns as `eicu_setC_full.csv` after filtering out patients with `prior_seizures = 1`, `pre_admission_aed = 1`, or `mechanical_ventilation = 1`. This is the "pure post-craniotomy" sensitivity stratum.

## `nis_chronic.parquet` · 2,518 × 50

NIS 2016–2019 chronic-SDH + surgical cohort with the corrected outcome definition. Columns:

- Demographics: `AGE`, `female`.
- Race indicators: `race_white`, `race_black`, `race_hispanic`, `race_asian`, `race_other`.
- Payer indicators: `pay_medicare`, `pay_medicaid`, `pay_private`, `pay_other`.
- Income quartile: `income_q1`–`income_q4`.
- Admission: `elective`, `weekend`, `transfer_in`.
- Procedure: `proc_craniotomy`, `proc_burr_hole`.
- SDH subtype: `is_chronic`, `is_acute`.
- Severity: `n_dx`, `n_pr`, `comorbidity_count`.
- Hospital region: `div_1`–`div_9`.
- Comorbidities: `hypertension`, `diabetes`, `afib`, `heart_failure`, `ckd`, `coagulopathy`, `alcohol`, `liver_disease`, `dementia`, `obesity`, `tobacco`, `prior_stroke`, `cad`.
- Antithrombotics: `antiplatelet`, `anticoagulant`.
- **Outcomes**: `seizure_acute` (corrected acute symptomatic, R56.x / 780.39 / G41.x only); `seizure_combined` (original conflated outcome, included for direct comparison).

## `icd10_corrected_outcome.csv` · 26 × 4

The released codeset distinguishing the two outcome families. Columns: `code`, `family` (acute_symptomatic / pre_existing_epilepsy), `description`, `included_in_corrected_outcome` (Yes / No). This is the canonical reference for any future NIS reanalysis.

## `firth_coefficients.csv` · 22 × 4

Firth penalized logistic regression coefficient estimates from the BIDMC postoperative-A fit. Columns: `variable`, `beta`, `ci_lo`, `ci_hi`. Reviewers may use these to verify the deployment model independently of the cross-validation pipeline.

## `SHA256SUMS.txt`

One row per file; format `<sha256>  <relative_path>`. Generated by `sha256sum` on Linux / `shasum -a 256` on macOS at the time of bundle preparation.
