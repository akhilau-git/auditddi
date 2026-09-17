# AuditDDI Paper Run Protocol

## Frozen manuscript title

**AuditDDI : AuditDDI-An Auditable Graph-Fingerpront Framework for Cold-Start Drug-Drug Interaction Prediction**

This protocol freezes the evidence-generation workflow. It does not authorise
clinical claims: the task distinguishes reported interactions from sampled
unreported pairs, which are not known-safe pairs.

## Split definitions used by this project

- **Transductive:** both drugs occur in the training drug set.
- **S1 cold start:** neither drug occurs in the training drug set.
- **S2 semi-cold start:** exactly one drug occurs in the training drug set.
- **Scaffold-disjoint:** train, validation, and test use disjoint Bemis-Murcko
  scaffold roles. This is a separate evaluation and must not be pooled with
  Transductive/S1/S2 results.

Use these definitions verbatim in the manuscript, figures, and tables.

## Required paper evidence

1. Run the `paper` preset with the five matched model seeds `11,23,37,53,71`.
2. Retain the ECFP logistic, legacy GAT, and edge-aware GAT comparison. Do not
   select a model using the held-out test partitions.
3. Archive each run's manifest, split CSVs, prediction CSVs, metrics JSON/CSV,
   figures, training history, source revision, and checkpoint SHA-256.
4. Run the scaffold-disjoint study separately and report it as a distinct
   experiment.
5. Freeze the selected checkpoint and run `evaluate_external_dataset.py` only
   with an independently sourced, provenance-described dataset. The external
   dataset must not be used for tuning or checkpoint selection.
6. Run normal offline tests with `pytest -q`. Live UniProt tests are deliberately
   excluded because they contact a third-party service; run them in Colab with
   `pytest -m live_network` after confirming internet access.

## Required reporting

Report mean and 95% confidence intervals across the five matched seeds for
AUROC, PR-AUC, MCC, Brier score, and ECE. Include seed-level values and raw
predictions as supplementary material. Do not report a best seed as the final
result.

## Submission gate

The manuscript may be submitted only after the final artifacts are archived,
all offline tests pass, external evaluation is completed or transparently
unavailable, and every claim is limited to the evidence produced by this
protocol.
