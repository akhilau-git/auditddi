# AuditDDI research roadmap status

This status distinguishes source implementation from Colab evidence. Datasets are expected in Google Drive and training is run in Colab. The local archive contains selected outputs, not the raw Drive snapshot.

| Workstream | Current evidence | Remaining work |
|---|---|---|
| Correctness and reproducibility | Current training and evaluation source includes split-aware unreported-negative sampling, separate model-selection and post-hoc validation roles, manifests, split hashes, prediction exports, and tests. | Re-run the release verification suite on the chosen revision and preserve the output. Do not mix legacy artifacts with current protocol. |
| S1/S2 screening | Archived GPU screening run `backend/checkpoints/auditddi_final_screening_v1/edge_aware_multitask/seed_11/artifacts/run_20260921T231616Z/`: one model seed. AUROC 0.9161 transductive (n=18,120; test-bootstrap 95% CI 0.9124–0.9201), 0.6698 S2 (n=15,086; 0.6615–0.6778), and 0.5003 S1 (n=620; 0.4571–0.5480). | Complete matched seeds on identical valid split hashes. S1 remains near chance; do not claim successful S1 generalization. |
| Calibration and abstention | Seed-11 run reports ECE 0.0513 transductive, 0.1581 S2, 0.2631 S1. Nominal-90% conformal observed coverage: 91.6% transductive, 72.3% S2, 62.6% S1. | Cold-start coverage misses target. Study score calibration, conformal nonconformity, shift-aware alternatives, retained error, and coverage on repeated runs. Do not describe nominal validity as observed guarantee under shift. |
| Paper baselines and ablations | `backend/checkpoints/paper_experiments/paper_benchmark` has partial outputs. Its current plan requests one seed; `study_summary.json` contains null comparison metrics. | Run the paper preset with matched five seeds 11, 23, 37, 53, 71 and identical verified data/splits. Complete chemical baseline and matched ablations before manuscript claims. |
| Invalid historical study | `backend/checkpoints/multimodal_seed_study` lists five seeds, but its saved summary has identical S1 dev/test hashes and four-row S1 test partitions. | Exclude from confirmatory reporting. Regenerate valid splits and rerun if useful. |
| Scaffold OOD | Historical report: AUROC 0.5537 vs 0.5393; p=5.5011e-7. Raw paired artifacts were not located in the inspected local archive. | Recover manifests/predictions or rerun, and report absolute performance, effect size, and uncertainty. Statistical significance alone does not establish utility. |
| Independent validation | Generic external evaluation tooling and provenance checks are implemented. No independent DrugBank/DDInter result was found in the local artifacts inspected. | Obtain/prepare a legally accessible dataset in Drive; record source, labels, dates, mapping attrition, pair overlap, and any shared compounds. Keep it out of tuning. |
| Temporal and leave-one-drug-out studies | No executed result found in the local artifacts inspected. | Define time-stamped label cohort and leave-one-drug-out protocol, then run in Colab if the source data support valid temporal metadata and sufficient labels. |
| Modality coverage and causal plausibility | Feature pipelines and audit hooks exist. | Produce source-wise coverage/missingness and intersection counts from the exact benchmark snapshot; conduct curated mechanism-recovery evaluation before mechanistic claims. |
| Deployment | Legacy research checkpoints are present. New candidate promotion is explicitly prohibited by the study plan. | Review a completed, repeated-seed candidate and provenance before manual checkpoint promotion. Keep the API research-only. |

## Dataset interpretation limits

- A sampled unreported TWOSIDES pair is not a known-safe pair.
- FAERS signals are observational and full-history leakage risk remains; 58 conflicting mapped structures are excluded in the archived run.
- A source pipeline existing in the repository does not mean that source was used by a candidate. Report actual input channels and mapping coverage from that run manifest.
- Morgan ECFP is a standard fingerprint, not a novel graph fingerprint.
- Explanations, attention, calibration, conformal sets, OOD flags, and abstention are research diagnostics, not clinical evidence.
- No candidate may overwrite the deployed checkpoint without a documented review.
