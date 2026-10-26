# Pillar 2 — Single-drug FAERS severity

Estimate the fraction of eligible FAERS case reports for one drug with a recorded serious outcome. This is a report fraction, not clinical incidence or causal toxicity risk.

## Workflow

- **Code:** local repository `D:\AuditDDI`, pushed to the GitHub repository; Colab pulls the selected commit.
- **Data and outputs:** Google Drive data root `MyDrive/Projects/auditddi`, mounted in Colab. Keep raw quarterly FAERS archives, bridge tables, labels, and results in this data root.
- **Preparation and training:** Colab is the active environment for both. Kaggle instructions are historical and are not the current workflow.
- **Results:** `results/pillars/single_drug_toxicity/` under the Drive data root.
- **Implementation:** `scripts/build_faers_multiquarter_toxicity_labels.py` prepares the 2025 train / 2026 H1 temporal cohort; `scripts/train_single_drug_toxicity_beta_binomial.py` is the existing exploratory count-aware model.

The one-quarter ExtraTrees and beta-binomial pilots are complete but did not reliably beat a training-mean baseline. The multi-quarter cohort and temporal evaluation remain pending until the label build succeeds and its manifest is reviewed.
