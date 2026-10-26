# Pillar 1 — Drug–drug interaction prediction

Predict reported TWOSIDES pair labels. DDInter remains separately tagged curated evidence and a negative-sampling exclusion source; its absence means unknown.

## Workflow

- **Colab preparation and training:** `notebooks/01_DDI_prediction.ipynb` uses the Drive-prepared data, audits required inputs, and runs the strict-identity pilot.
- **Pilot models:** Morgan-fingerprint logistic regression, edge-aware GATv2, and the seven-input DDI model. The neural models are configured for up to 200 epochs; the first run uses seed 42.
- **Drive data:** `MyDrive/Projects/auditddi` (Colab mount path may vary; the notebook checks the common mount locations).
- **Results:** `Projects/auditddi/results/pillars/DDI/`.
- **Implementation:** `src/training/run_experiment_suite.py`, `src/training/train_full_pipeline_v2.py`, and `src/models/seven_branch_ddi.py`.

Edit locally in `D:\AuditDDI`, push the intended code revision to GitHub, then pull it into Colab. The code checkout and the Drive data folder are separate.

Training loads the frozen audited split CSVs directly and validates cohort, exclusions, split seed, source-file hashes, row counts, and pair uniqueness against the saved split manifest.
