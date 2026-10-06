# AuditDDI Model and Evidence Card

## Intended use

AuditDDI is a research prototype for binary classification of reported TWOSIDES drug pairs against sampled unreported pairs. Sampled unreported pairs are not confirmed safe combinations. The system is not validated for clinical decision-making, patient-specific prescribing, triage, or diagnosis; it does not recommend treatment changes.

## Current local serving checkpoint

`backend/checkpoints/auditddi_model.pt` is a legacy research checkpoint. The API identifies it as uncalibrated and does not treat its stored validation score as test or clinical evidence. Do not promote a Colab candidate into this path without a separate documented review.

## Archived candidate evaluation

The local artifact archive contains `backend/checkpoints/auditddi_final_screening_v1/edge_aware_multitask/seed_11/artifacts/run_20260921T231616Z/`, an edge-aware multitask Colab T4 screening run (one model seed, split seed 42). It evaluated 18,120 transductive, 15,086 S2, and 620 S1 pairs. The score report is in `results_summary.json`; raw predictions, split files, manifests, and training history are stored alongside it.

| Split | AUROC | Test-row bootstrap 95% CI | AUPRC | Brier | ECE | Observed conformal coverage / abstention |
|---|---:|---:|---:|---:|---:|---:|
| Transductive | 0.9161 | 0.9124–0.9201 | 0.9294 | 0.1172 | 0.0513 | 91.6% / 19.5% |
| S1, both drugs unseen | 0.5003 | 0.4571–0.5480 | 0.5116 | 0.3446 | 0.2631 | 62.6% / 23.5% |
| S2, one drug unseen | 0.6698 | 0.6615–0.6778 | 0.6880 | 0.2575 | 0.1581 | 72.3% / 21.5% |

Conformal target coverage was nominally 90%. Observed coverage on S1 and S2 is below target; therefore the current rule has not shown its nominal coverage under cold-start distribution shift. S1 ranking is near chance. Intervals are bootstrap intervals over test rows for one fitted seed; they do not estimate variation across training seeds.

## Data, features, and naming

The source tree contains RDKit graph and standard Morgan fingerprint code plus optional PharmGKB, BindingDB, UniProt, PDB, GEO, FAERS, ChEMBL, and PubChem data pipelines. These are not all simultaneously active model modalities. Use a run's manifest for its actual feature channels and source coverage. The project does not introduce a novel graph fingerprint or structural hash; Morgan ECFP is an established representation.

The archived run uses TWOSIDES-derived reported interactions and split-aware sampled unreported pairs. FAERS toxicity features are governed by the separate auxiliary-label protocol and are not evidence of causal toxicity. The run excluded 58 conflicting canonical structures from its toxicity bridge. No independent DrugBank/DDInter benchmark result was found in the local archive inspected.

## Calibration, abstention, explanations

Calibration, split-conformal, structural applicability, and atom/motif attribution utilities are present. Their usefulness depends on validation for the candidate, split, and distribution. Cold-start ECE and observed coverage in the archived run indicate substantial degradation. Explanations are model sensitivity evidence, not validated mechanisms. None of these methods establish clinical risk or safety.

## Evidence gaps

- Complete a valid matched five-seed benchmark on fixed split and data hashes. The current paper benchmark summary has null comparison metrics and its plan requests one seed.
- Exclude the archived `multimodal_seed_study` summary from confirmatory claims: S1 development/test split hashes match and the S1 test has four rows.
- Recover raw artifacts or rerun the historical scaffold study; report absolute effect and uncertainty.
- Run independent DrugBank/DDInter evaluation, temporal and leave-one-drug-out evaluation where source dates/labels support them, modality coverage audits, and matched ablations.
- Reassess calibration and conformal coverage under cold-start shift and report selective error with coverage.

## Reproduction

Model training and dataset access occur in Google Colab using data on Google Drive. The local checkout contains selected outputs but not the raw Drive data snapshot. Preserve the run manifest, source revision, dataset and split hashes, predictions, metrics, and checkpoint digest for any new result. Do not treat a single seed as publication-grade repeated-seed evidence.
