# Pillar 1 — Drug–drug interaction prediction

Predict reported TWOSIDES pair labels. DDInter remains separately tagged curated evidence and a negative-sampling exclusion source; its absence means unknown.

## Workflow

- **Colab preparation and training:** `notebooks/01_DDI_prediction.ipynb` uses the Drive-prepared data, audits required inputs, and runs the strict-identity comparison.
- **Pilot models:** Morgan-fingerprint logistic regression, edge-aware GATv2, and the seven-input DDI model. The neural models are configured for up to 200 epochs; the first run uses seed 42.
- **Cold-start selection:** the next comparison selects checkpoints on the pair-disjoint S1-dev selection subset and fits calibration/threshold roles on its reserved post-hoc subset. S1-test remains untouched until final evaluation.
- **Optional runtime-heavy audit:** pair-level Tanimoto novelty scoring can be disabled for the primary score run; it does not change model predictions or headline metrics.
- **Drive data:** `MyDrive/Projects/auditddi` (Colab mount path may vary; the notebook checks the common mount locations).
- **Results:** `Projects/auditddi/results/pillars/DDI/`.
- **Implementation:** `src/training/run_experiment_suite.py`, `src/training/train_full_pipeline_v2.py`, and `src/models/seven_branch_ddi.py`.

Edit locally in `D:\AuditDDI`, push the intended code revision to GitHub, then pull it into Colab. The code checkout and the Drive data folder are separate.

Training loads the frozen audited split CSVs directly and validates cohort, exclusions, split seed, source-file hashes, row counts, and pair uniqueness against the saved split manifest.

The earlier edge-aware GATv2 run achieved transductive AUROC 0.894 but S1 AUROC 0.529 and S2 AUROC 0.664. It was a graph-only model selected on ordinary validation, not the proposed seven-input model. The saved headline metric CSV is available, but its evaluation process was interrupted during post-evaluation audits. The next controlled run uses the cold-start selection protocol above and compares all three models on the same frozen test splits.
