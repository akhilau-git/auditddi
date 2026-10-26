# Single-drug FAERS severity — final current results

**Status:** the current modeling comparison is complete. The selected model is a five-seed Random Forest ensemble. Results below are from the Colab outputs shared for this project; they are research results for a FAERS report statistic, not clinical toxicity-risk estimates.

## Target and cohort

For each mapped drug structure, the target is the number of eligible FAERS case reports with a recorded serious outcome divided by the total eligible reports for that structure. The builder used suspect drug roles (`PS` and `SS`) and serious outcome codes `CA`, `DE`, `DS`, `HO`, `LT`, `OT`, and `RI`.

- Training period: 2025 Q1–Q4, 1,175 structures and 1,197,978 mapped case-structure reports.
- Temporal test period: new cases from 2026 Q1–Q2, 1,157 structures and 546,515 mapped case-structure reports.
- Train/test CASEID overlap: zero.
- Test structures also present in training: 1,149; test-only structures: 8.
- These counts support a temporal evaluation mainly on known structures. Eight new structures are too few for a dependable cold-start conclusion.

## Final model choice

Use a **five-seed Random Forest prediction average** as the single selected model for this project iteration. The chosen settings were 500 trees, bootstrap sampling, `max_features=0.8`, and `min_samples_leaf=3`; seeds were 203–207. Model settings were selected using five-fold scaffold-grouped validation on 2025 data only.

| Model | 2025 scaffold CV MAE (mean ± fold SD) | 2026 H1 MAE | RMSE | R² | Spearman | sqrt-report-weighted MAE |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest, five-seed mean** | **0.10843 ± 0.00214** | **0.08598** | **0.14186** | **0.42116** | **0.61875** | 0.08277 |
| ExtraTrees, five-seed mean | 0.11448 ± 0.00360 | 0.08661 | 0.14434 | 0.40081 | 0.60828 | **0.08221** |
| XGBoost, five-seed mean | 0.11151 ± 0.00210 | 0.10907 | 0.16479 | 0.21895 | 0.34751 | 0.10861 |
| Train-mean baseline | — | 0.12527 | 0.18665 | -0.00197 | — | 0.13824 |

Random Forest had the best scaffold-validation MAE and slightly better temporal-test MAE, RMSE, R², and Spearman than ExtraTrees. ExtraTrees had slightly better report-weighted MAE, so the model choice is based on **per-drug MAE as the primary metric**. The unweighted temporal-test MAE difference between Random Forest and ExtraTrees is small; treat Random Forest as the selected candidate, not a proven universal winner.

Other tested approaches were not selected: frozen ChemBERTa plus Ridge (test MAE 0.11930), beta-binomial MLP (five-seed mean MAE 0.14876), XGBoost, and the fixed 50:50 ExtraTrees/ChemBERTa blend (MAE 0.10055). The 50:50 blend was worse than ExtraTrees alone.

## Metrics and interpretation

This is a **regression** task: the model predicts a continuous report fraction. Accuracy is not applicable unless a meaningful category threshold is defined in advance. Report MAE, RMSE, R², Spearman correlation, and the report-count-weighted MAE instead.

The target is a spontaneous-reporting statistic. FAERS does not provide the exposed patient population needed to estimate incidence, and report counts can be affected by reporting bias, drug use, indication, and other factors. Do not describe the model output as a patient's probability of toxicity or as causal drug toxicity.

## Reproducible output locations

All data and results are in Google Drive under `MyDrive/Projects/auditddi`.

- Cohort labels and manifest: `results/pillars/single_drug_toxicity/label_builds/faers_2025_train_2026h1_test_20261026_180851/`
- Tree model comparison and saved seed models: `results/pillars/single_drug_toxicity/tree_ensemble_comparisons/ecfp6_scaffold_cv_20261026_192012/`
- Five-seed ExtraTrees reference: `results/pillars/single_drug_toxicity/temporal_models/faers_2025_train_2026h1_test_five_seed_20261026_183858/`
- Frozen ChemBERTa comparison: `results/pillars/single_drug_toxicity/pretrained_models/chemberta_frozen_20261026_185436/`

The 2026 H1 test results have been inspected while comparing models and are therefore exploratory for model selection. For independent confirmation, freeze the Random Forest settings and evaluate them on a later FAERS period that has not been used for tuning or model choice.
