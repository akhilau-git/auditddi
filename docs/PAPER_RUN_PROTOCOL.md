# AuditDDI Paper Run Protocol

## Manuscript title

**AuditDDI: An Auditable Framework for Drug-Drug Interaction Prediction under Cold-Start Evaluation**

This title describes the evaluation question and does not imply that cold-start performance has been solved. The implementation uses standard RDKit Morgan fingerprints; it does not introduce a novel graph fingerprint. Do not claim a fingerprint innovation.

## Split definitions

- **Transductive:** both drugs occur in the training drug set.
- **S1 cold start:** neither drug occurs in the training drug set.
- **S2 semi-cold start:** exactly one drug occurs in the training drug set.
- **Scaffold-disjoint:** train, validation, and test use disjoint Bemis-Murcko scaffold roles. This is a separate evaluation and must not be pooled with Transductive/S1/S2 results.

A sampled unreported TWOSIDES pair is not a known-safe pair. S1/S2 scores must be reported whether strong or weak; do not infer successful generalization from the split design alone.

## Required evidence

1. Run the `paper` preset with the five matched model seeds `11,23,37,53,71`, after confirming the same source snapshot, master-node hash, and split hashes.
2. Retain ECFP logistic, legacy GAT, and edge-aware GAT baselines. Select candidates using development partitions only; evaluate test partitions once.
3. Archive run manifests, split and prediction CSVs, metrics, figures, training history, source revision, and checkpoint SHA-256.
4. Keep scaffold-disjoint evaluation separate. Report effect size and confidence interval alongside any p-value.
5. Evaluate a genuinely independent dataset only with source, label definition, temporal independence, identifier mapping, and overlap documented. Do not tune on it.
6. Report calibration, conformal coverage, abstention rate, and retained-subset error as measured diagnostics. Do not present nominal conformal coverage as empirically verified coverage.
7. Run `pytest -q` as the project's verification step before release. Live UniProt tests are excluded by default and require network-enabled Colab.

## Reporting

Report seed-level values and mean with 95% confidence intervals for AUROC, AUPRC, MCC, Brier score, and ECE, together with class prevalence and the negative sampling definition. Report source coverage/missingness and exact input channels used. Do not report only the best seed.

## Submission gate

Submit only after the frozen runs and artifacts are archived, offline tests pass, external evaluation is complete or explicitly unavailable, and claims are limited to produced evidence. Until then the manuscript remains a protocol and preliminary historical report.
