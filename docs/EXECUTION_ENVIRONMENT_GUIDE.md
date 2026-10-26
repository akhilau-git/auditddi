# Execution environment guide

## Current project workflow

Use Colab for dataset preparation, audits, and model training. The user has Colab Pro and has chosen Colab for the DDI training run, so Kaggle is not required for this run. A private Kaggle input dataset created during setup can remain unused; the authoritative data stay in Google Drive.

| Work | Environment | Code | Data and outputs |
|---|---|---|---|
| Dataset collection, identity matching, feature preparation, audits, and model training | Colab | Pull the selected revision from `https://github.com/akhilau-git/auditddi.git` | `/content/drive/MyDrive/Projects/auditddi` or the active Colab mount path |
| Local code edits and Git operations | Windows local machine | `D:\AuditDDI` | Do not use this as the dataset root |

The GitHub checkout and Drive project folder are different. Edit code in `D:\AuditDDI`, push the intended code revision to GitHub, then clone or pull that revision in Colab. Do not upload a source ZIP as the normal workflow. Store pillar outputs under `Projects/auditddi/results/pillars/<pillar>/`.

## DDI Pillar 1

- Colab notebook: `pillars/ddi/notebooks/01_DDI_prediction.ipynb`.
- Data root: `Projects/auditddi` in Google Drive.
- The first run is a strict-identity pilot with seed 42 and up to 200 epochs for the neural models. It compares Morgan logistic regression, edge-aware GATv2, and the seven-input DDI model.
- The seven-input model requires the prepared feature bundle and the ChEMBL-pretrained encoder checkpoint.
- The trainer currently regenerates splits with seed 42. Compare the run's split manifest with the fixed audited splits in Drive before treating metrics as final.
- Results are written under `Projects/auditddi/results/pillars/DDI/`.

## Other pillars

Keep DDI classification, single-drug FAERS severity, drug-pair FAERS severity, and ADMET screening as separate tasks with separate labels and outputs. When those model-training pillars are resumed, use Colab for training under the current environment choice. FAERS report fractions describe submitted reports, not incidence or causality.
