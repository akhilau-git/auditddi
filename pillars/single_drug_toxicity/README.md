# Pillar 2 — Single-drug FAERS severity

Estimate the fraction of eligible FAERS case reports for one drug with a recorded serious outcome. This is a report fraction, not clinical incidence or causal toxicity risk.

## Workflow

- **Code:** local repository `D:\AuditDDI`, pushed to the GitHub repository; Colab pulls the selected commit.
- **Data and outputs:** Google Drive data root `MyDrive/Projects/auditddi`, mounted in Colab. Keep raw quarterly FAERS archives, bridge tables, labels, and results in this data root.
- **Preparation and training:** Colab is the active environment for both. Kaggle instructions are historical and are not the current workflow.
- **Results:** `results/pillars/single_drug_toxicity/` under the Drive data root.
- **Implementation:** `scripts/build_faers_multiquarter_toxicity_labels.py` prepares the 2025 train / 2026 H1 temporal cohort; `scripts/train_single_drug_toxicity_temporal.py` compares training-only mean baselines, Morgan ExtraTrees, and a count-aware beta-binomial MLP; `scripts/train_single_drug_toxicity_chemberta.py` tests a frozen ChemBERTa encoder with a Ridge regression head, selecting Ridge strength on a 2025 scaffold-disjoint validation split; `scripts/compare_single_drug_tree_ensembles.py` compares bootstrap Random Forest and XGBoost with Morgan ExtraTrees using five-fold scaffold validation on 2025 labels. Colab notebooks are under `pillars/single_drug_toxicity/notebooks/`.

The one-quarter ExtraTrees and beta-binomial pilots did not reliably beat a training-mean baseline. The multi-quarter cohort has now been built and audited. Its temporal test shares most structures with training, so it evaluates later reports for mostly known drugs; it is not a robust cold-start drug test. All outputs remain descriptive FAERS report statistics, not clinical toxicity risks.

The five-seed temporal comparison found Morgan ExtraTrees substantially better than the train-mean baseline on the observed 2026 H1 test cohort, while the beta-binomial MLP did not help. This same test cohort has been inspected, so later model comparisons on it are exploratory. Use a later untouched FAERS period for independent confirmation.
