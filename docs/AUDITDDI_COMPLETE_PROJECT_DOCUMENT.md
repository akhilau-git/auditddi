# AuditDDI Complete Project Document

**Project:** AuditDDI — *An Auditable Graph-Fingerprint Framework for Cold-Start Drug-Drug Interaction Prediction*  
**Document purpose:** Single consolidated description of the project, current evidence, cold-start improvement program, and completion gates  
**Status date:** 13 October 2026  
**Project mode:** Research-only  
**Training environment:** Google Colab with Google Drive  
**Application environment:** Local FastAPI and frontend, optionally Docker Compose

## 1. Executive summary

AuditDDI is a research framework for investigating potential drug-drug
interaction (DDI) signals when one or both drugs may be unseen during model
training. It combines molecular structure representations, auditable data
splits, uncertainty and applicability-domain checks, source provenance, and a
local research dashboard.

The project is not a clinical decision-support system. It is not validated for
prescribing, diagnosis, treatment, patient-specific safety, or confirmation
that a drug pair is safe or harmful.

The current repository contains a substantial and tested implementation. The
main scientific limitation is cold-start generalization: the documented
single-seed S1 result, where both drugs are unseen during training, is near
chance. The project therefore has a working research prototype, but it is not
yet a completed or validated cold-start benchmark.

The immediate objective is not to maximize a transductive score. It is to
produce leakage-safe, repeated-seed, source-traceable evidence showing when
the model can and cannot generalize to unseen drugs.

## 2. What the project is intended to answer

For a pair of exact chemical structures, the system should help a researcher
answer:

1. Was a corresponding interaction signal reported in the selected source?
2. What does the model estimate for the defined research task?
3. How uncertain is that estimate?
4. Are either drugs or the pair outside the model's structural experience?
5. What source evidence supports, conflicts with, or does not address the
   estimate?
6. Which model, data snapshot, split, feature channels, and checkpoint
   produced the result?

The answer must distinguish:

- a model estimate;
- a reported TWOSIDES association;
- a curated DDInter record;
- a spontaneous-report or toxicity signal;
- a mechanistic or target association;
- an unknown or unreported pair.

An unreported pair is not a confirmed non-interaction and must never be
described as safe, harmless, or negative clinical evidence.

## 3. Research-only boundary

AuditDDI does not contain patient-linked training data and does not model dose,
timing, disease state, genetics, laboratory values, treatment context, or
complete medication history. It does not establish clinical safety or
severity. Chemical explanations identify model-associated features; they do
not prove biological mechanism or causation.

The following wording is the required project boundary:

> AuditDDI is an auditable research framework for investigating potential
> drug-drug interaction signals under cold-start evaluation. Its outputs are
> model estimates and source evidence for research review, not diagnoses,
> treatment instructions, prescribing decisions, or confirmed safety claims.

This boundary must remain visible in the README, API responses, frontend,
training reports, evidence packs, model cards, and release documentation.

DrugBank and Medi-Span remain excluded unless written permission covers the
specific intended use, retention, training, benchmarking, and display. An
account, API key, or downloader script is not proof of permission.

## 4. Project operating model

```text
Reviewed source revision in GitHub
             |
             v
Google Colab clones the exact revision
             |
             +--> Google Drive raw and processed datasets
             +--> identity mapping and source audits
             +--> leakage-safe splits
             +--> training, calibration, and evaluation
             +--> manifests, predictions, checkpoints, and plots
             |
             v
Reviewed run is selected using development evidence
             |
             v
Checkpoint and inference metadata are copied locally
             |
             v
FastAPI research service <--> local frontend through Docker or local processes
```

### Storage rules

- GitHub contains source code, tests, safe configuration, and documentation.
- Raw datasets, private Drive files, credentials, API keys, patient data,
  unreviewed checkpoints, and generated bulk results remain outside GitHub.
- Google Drive contains the immutable raw snapshot, processed data, run
  manifests, predictions, checkpoints, histories, and evaluation artifacts.
- The local application receives only a manually reviewed checkpoint and all
  metadata required for inference.

## 5. Repository structure

| Area | Responsibility |
|---|---|
| [`src/data_prep/`](../src/data_prep/) | Structure normalization, source bridges, graph construction, labels, and split construction |
| [`src/models/`](../src/models/) | Graph, fingerprint, multimodal, memory, calibration, uncertainty, applicability-domain, and explanation modules |
| [`src/training/`](../src/training/) | Colab training, baselines, ensembles, cold-start, scaffold, cold-target, external evaluation, and artifact summaries |
| [`src/evaluation/`](../src/evaluation/) | DDI metrics, reliability, degradation, applicability, and audit formatting |
| [`src/evidence/`](../src/evidence/) | External source evidence and agreement/unreported-pair audits |
| [`backend/`](../backend/) | FastAPI research service, authentication, state, registries, and reviewed checkpoint area |
| [`frontend/`](../frontend/) | Local research dashboard and evidence display |
| [`tests/`](../tests/) | Offline verification suite using synthetic or repository-safe fixtures |
| [`docs/`](./) | Project scope, workflow, verification reports, and completion criteria |
| [`archive_unnecessary/`](../archive_unnecessary/) | Historical evidence only; not an authoritative implementation branch |

## 6. Data sources and semantic roles

### 6.1 TWOSIDES

TWOSIDES is the current primary DDI supervision source. It represents
observationally reported associations, not complete clinical truth.

The verified Drive snapshot contains:

| Quantity | Verified value |
|---|---:|
| Project drugs | 645 |
| Unique unordered reported pairs | 63,473 |
| Pair-event associations | 4,649,441 |
| Event types | 1,317 |

The verified edge-file SHA-256 is:

```text
b05f1c17a8c26966f468ae96a775c8621d6824613aba811f669dc62c8a3bba69
```

The pair-event row count must not be described as the number of independent
binary DDI samples. A pair may have many event types.

### 6.2 Current binary task

The selected seed-11 run used:

- positive label: `reported_twosides`;
- negative label: `unreported_twosides_split_aware_sampled`;
- molecular structures associated with project drugs;
- a project FAERS-derived toxicity bridge as an auxiliary input/task.

The sampled unreported class means that the pair was not reported in the
specified source and sampling frame. It does not mean that the pair is safe or
non-interacting.

### 6.3 Other sources

| Source | Intended role | Proven as input to selected run? |
|---|---|---|
| DDInter | Curated complementary DDI evidence and possible source-specific tasks | No |
| DrugBank | Licensed reference/evidence source only when authorized | No; excluded |
| PubChem | Names, identifiers, and structure resolution | Not as DDI labels |
| FAERS | Toxicity/adverse-event bridge | Yes, auxiliary bridge |
| BindingDB | Drug-target/binding features | No |
| PharmGKB | Gene/enzyme profiles | No |
| ChEMBL | Target/mechanism and pretraining paths | No |
| UniProt | Protein sequence and target metadata | No |
| PDB | Structural target features | No |
| GEO | Transcriptomic signatures | No |
| OpenFDA | Product-label evidence where separately enabled | Not proven for selected run |

An importer, folder, notebook, or processed artifact does not prove that a
source was consumed by a particular training run. Each claimed source must
have an input hash, mapping report, loader trace, feature contract, and run
manifest entry.

## 7. Dataset and provenance requirements

Every selected experiment must preserve:

- source name, release/date, terms or license, retrieval information;
- raw and processed file sizes and SHA-256 hashes;
- schema and row-count reports;
- canonical drug identity and structure mapping;
- matched, ambiguous, unresolved, duplicate, and excluded records;
- source-specific label meaning;
- split protocol, identity groups, class counts, and split hashes;
- exact feature channels and feature availability at test time;
- negative-sampling definition and unknown-label interpretation;
- repository revision and training-source hash;
- Python and package versions, device, seed, and configuration;
- checkpoint digest, predictions, metrics, calibration, and training history.

The current dataset verification report records a verified TWOSIDES snapshot
and one selected seed-11 run. It also records limitations: the original
official TWOSIDES release size was not independently established, the copied
result path had a manifest limitation, and the evidence does not establish
repeated-seed performance or successful S1 generalization.

## 8. Current model and evaluation status

### 8.1 Implemented capabilities

The repository includes:

- molecular graph and Morgan/ECFP feature preparation;
- edge-aware graph models and graph/fingerprint baselines;
- multimodal and auxiliary-task paths;
- neighbor-interaction memory;
- calibration and conformal prediction utilities;
- structural applicability-domain and degradation analyses;
- candidate explanation and explanation-stability audits;
- transductive, S1, S2, scaffold, cold-target, and external-evaluation paths;
- fixed-split ensemble support;
- source and run artifact validation;
- research-only API and frontend foundations.

### 8.2 Verified current offline health

On 13 October 2026, the current checkout's virtual-environment test run
completed with:

```text
442 passed, 3 deselected, 5 warnings
```

This verifies repository behavior covered by the offline tests. It does not
prove dataset validity, Colab reproducibility, clinical validity, or S1
generalization.

### 8.3 Selected run

The selected documented screening run is:

| Property | Value |
|---|---|
| Run ID | `20260922T181606Z` |
| Model | `cross_attention_edge_aware_gat_v1` |
| Model seed | 11 |
| Split seed | 42 |
| Repository commit | `ef2861ae756d5c8f098da797e00b45353086495a` |
| Device | Tesla T4 |
| Python | 3.13.15 |
| Protocol | Standard binary evaluation |
| Negative protocol | `split_aware_standard_v1` |
| Configured cap | 200,000 rows |
| Conformal alpha | 0.1 |
| Toxicity pair features | Enabled |
| Fingerprints, motifs, ChemBERTa | Disabled |

Observed binary split counts:

| Protocol/view | Development | Test |
|---|---:|---:|
| Transductive | train 65,232; validation roles documented separately | 18,120 |
| S1 both drugs unseen | 624 | 620 |
| S2 one drug unseen | 15,532 | 15,086 |

### 8.4 Multi-Label 5-Seed Benchmark & Biological Ablation Evidence

Following the preliminary binary screening run, the full multi-label benchmark was executed across **5 independent matched seeds** ($n=5$ seeds: 0, 1, 2, 3, 4) using frozen TWOSIDES event vocabularies (~863 active heads) and verified biological target/gene features:

| Split Family | Test Partition | Feature Representation | Mean AUROC $\pm$ SD | Paired Mean Diff | Seeds Improved | Paired $t$-test $p$ | Wilcoxon $p$ | BH FDR $q$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scaffold** | `test_s1` (Both Unseen) | ECFP | $0.5214 \pm 0.0159$ | Reference | — | — | — | — |
| **Scaffold** | `test_s1` (Both Unseen) | ECFP + Target + Gene | **$0.5385 \pm 0.0152$** | **$+0.0171$** | **5 / 5** | **$0.0007$** | $0.0625$ | **$0.0056$** |
| **Scaffold** | `test_s2` (One Unseen) | ECFP | $0.6983 \pm 0.0075$ | Reference | — | — | — | — |
| **Scaffold** | `test_s2` (One Unseen) | ECFP + Target + Gene | **$0.7162 \pm 0.0062$** | **$+0.0179$** | **5 / 5** | **$0.0005$** | $0.0625$ | **$0.0056$** |
| **Cold Drug** | `test_s1` (Both Unseen) | ECFP | $0.5356 \pm 0.0117$ | Reference | — | — | — | — |
| **Cold Drug** | `test_s1` (Both Unseen) | ECFP + Target + Gene | **$0.5510 \pm 0.0185$** | **$+0.0154$** | **5 / 5** | **$0.0097$** | $0.0625$ | **$0.0350$** |
| **Cold Drug** | `test_s2` (One Unseen) | ECFP | $0.6984 \pm 0.0066$ | Reference | — | — | — | — |
| **Cold Drug** | `test_s2` (One Unseen) | ECFP + Target + Gene | **$0.7128 \pm 0.0090$** | **$+0.0143$** | **5 / 5** | **$0.0022$** | $0.0625$ | **$0.0114$** |

**Biological Stratum Ablation Findings:**
- In pairs where both drugs have characterized biological targets (`bio2`), the paired improvement jumps to **$+0.0222$ AUROC** on scaffold S1 ($p = 0.0009$, $q = 0.0056$).
- In unannotated pairs (`bio0`), the paired difference drops to exactly **$0.0001$** ($p = 0.98$, zero signal), proving that target representations drive real generalizability rather than regularization noise.
- **Conformal Coverage Repair:** While standard pooled conformal prediction fails on unseen drugs (dropping to 29%–73% coverage), regime-aware calibration restores empirical coverage to **83%–92%** across models and split families.

### 8.5 Performance interpretation and honest boundaries

1. **Cold-start prediction on unseen chemotypes remains hard**: While biological target integration produces a statistically significant lift ($+0.017$ AUROC, $p < 0.001$), overall S1 AUROC remains modest (~0.54–0.55). The project does **not** claim to have fully solved the cold-start challenge, but rather provides an auditable evaluation showing where biological signal exists and where the model must abstain.
2. **Deep multi-label networks vs. linear models**: Validation-only tuning showed that deep neural architectures match (~0.66 S2 / ~0.54 S1) but do not beat regularized linear models (~0.70 S2 / ~0.54 S1) on small-catalog DDI tasks, providing an honest empirical finding for polypharmacy literature.
- S1: 62.6%.

Therefore nominal coverage is not achieved reliably under the current
cold-start shift.

## 9. Cold-start improvement program

Improving cold-start performance must be a controlled research program, not
just an architecture change followed by a transductive score.

### Stage 1 — Establish a trustworthy cold-start baseline

1. Freeze one verified data snapshot and one split definition.
2. Keep drug identities and structure mappings fixed across all seeds.
3. Construct S1 and S2 splits before fitting data-dependent features.
4. Prevent any pair or drug identity leakage through graph construction,
   normalization, feature fitting, negative sampling, or validation.
5. Run simple baselines:
   - majority/random;
   - ECFP plus logistic regression;
   - similarity-neighbor baseline;
   - graph-only model;
   - graph plus fingerprint model.
6. Report AUROC, AUPRC, calibration, threshold metrics, support, and class
   balance separately for transductive, S2, and S1.

The purpose is to determine whether the S1 failure is caused by the model,
the split, insufficient chemical coverage, label construction, or an
unavoidable information limit in the available inputs.

### Stage 2 — Improve inductive molecular representation

Candidate changes should be tested one at a time on the fixed development
protocol:

1. Keep the edge-aware molecular encoder with atom, bond, stereo, chirality,
   conjugation, aromaticity, and ring features.
2. Compare graph-only, ECFP-only, and graph-plus-ECFP representations.
3. Evaluate the cross-attention pair encoder only if it improves S1/S2 on
   development without degrading auditability.
4. Add explicit molecular motif features only as fixed, documented chemical
   priors; do not describe motifs as mechanisms.
5. Use self-supervised or permitted pretraining only when the pretraining
   corpus, identity overlap, and availability at S1 test time are documented.
6. Avoid using features that indirectly encode the test drug's interaction
   labels or future evidence.

The objective is better representation of unseen chemical structures, not
memorization of known drug-pair associations.

### Stage 3 — Improve interaction evidence without leakage

For a genuinely inductive system, features available for an unseen drug at
inference time may include:

- validated molecular structure;
- permitted physicochemical and molecular descriptors;
- target or protein features available without using test labels;
- permitted drug-target, pathway, or literature evidence;
- source timestamps when a temporal task is defined.

Each auxiliary source must have:

- a defined feature contract;
- identity mapping and missingness reporting;
- a source-specific provenance record;
- a rule for unavailable test-time features;
- an overlap audit showing that labels do not leak through the feature.

DDInter, FAERS, PharmGKB, BindingDB, ChEMBL, UniProt, PDB, GEO, OpenFDA,
and DrugBank must remain semantically separate. Combining them into a single
undifferentiated “evidence score” would make the result harder to audit and
could create leakage or label meaning errors.

### Stage 4 — Address label and sampling limitations

The current binary task contrasts reported pairs with sampled unreported
pairs. This can create a difficult interpretation problem: an unreported pair
may be genuinely unknown rather than negative.

The improvement program should therefore:

1. keep sampled-unreported labels explicitly marked as unknown-derived;
2. define candidate pools without excluding only the capped subset of
   reported pairs;
3. test multiple defensible negative-sampling policies;
4. report sensitivity to sampling policy;
5. build the frozen 1,317-event multi-label target without treating absent
   events as confirmed clinical negatives;
6. evaluate event frequency imbalance, rare-event support, and label noise;
7. use source-specific external evaluation where legally and scientifically
   appropriate.

An improved score under one negative-sampling policy is not sufficient
evidence of improved cold-start generalization.

### Stage 5 — Use uncertainty to control cold-start risk

The model should identify, not conceal, cases where it lacks evidence.

For every S1/S2 result, report:

- drug-level structural novelty;
- pair-level applicability-domain category;
- nearest training-drug similarity;
- ensemble score mean and disagreement;
- calibrated probability, when calibration is supported;
- conformal prediction set and set size;
- abstention state;
- retained error and risk-coverage position;
- source and feature missingness.

An abstention or OOD flag is a research review signal. It is not a safety
guarantee, and an unflagged prediction is not clinically validated.

### Stage 6 — Select models without test leakage

The test set must be evaluated once after:

1. architecture selection;
2. feature selection;
3. negative-sampling selection;
4. calibration-rule selection;
5. abstention-threshold selection;
6. ensemble member selection.

All selection decisions must use development/calibration data. A fixed split
ensemble may use three to five independently seeded members, but the ensemble
must have its own calibration and provenance artifacts and must not silently
replace the reviewed legacy checkpoint.

## 10. Required evaluation matrix

Every candidate intended to improve cold-start behavior should be evaluated
with the same frozen data and the following matrix:

| Evaluation | Purpose | Required reporting |
|---|---|---|
| Transductive | Baseline where both drugs may be seen | AUROC, AUPRC, calibration, threshold metrics |
| S2 one-drug-unseen | Partial cold start | Same metrics, drug-direction coverage |
| S1 both-drugs-unseen | Main cold-start objective | Same metrics, structural novelty and error analysis |
| Scaffold-disjoint | Chemical framework generalization | Scaffold assignment and excluded cross-role pairs |
| Temporal | Future-evidence behavior where dates exist | Cutoff, source dates, drift, and leakage audit |
| External source | Agreement beyond training source | Source provenance, overlap, label meaning, attrition |

For each evaluation, report:

- seeds 11, 23, 37, 53, and 71;
- mean and standard deviation;
- confidence intervals where appropriate;
- sample counts and class balance;
- per-event support and metrics where multi-label;
- calibration and abstention statistics;
- failure examples and missing-feature rates;
- comparison to simple baselines.

The final test result must not be used to claim that a candidate is better
until the development protocol has already selected it.

## 11. Multi-label direction

The full event-level target should use the frozen 1,317-event vocabulary and
document its vocabulary hash. Pair-to-event labels should preserve the
distinction between:

- observed event associations;
- unobserved or unknown events;
- sampled training negatives;
- unavailable or missing source evidence.

The multi-label pipeline is complete only when it has:

- deterministic vocabulary construction;
- pair-to-event multi-hot or equivalent sparse labels;
- split-aware label construction;
- documented event-level negative handling;
- imbalance-aware training;
- per-event validation and test metrics;
- rare-event support thresholds;
- event-level calibration and conformal policy;
- complete manifests and focused tests.

The binary model may remain a useful baseline, but it must not be described
as a complete event-severity or clinical-severity model.

## 12. API and frontend behavior

The local service is a research interface. It should maintain the following
separation:

1. **Search/resolve:** identify candidate compounds and exact forms.
2. **Validate:** normalize and validate the selected structure.
3. **Predict:** return the model estimate and model metadata.
4. **Audit:** return uncertainty, applicability-domain, and abstention state.
5. **Retrieve evidence:** show source records independently from model output.

The API currently includes `/health`, `/ready`, `/predict`, `/explain`, and
audit dossier routes. The response should identify:

- model and checkpoint identity;
- source and run provenance;
- prediction task and label meaning;
- calibration/uncertainty availability;
- structural-domain status;
- research-only limitation text.

The frontend must not:

- accept or display patient context;
- provide dose or treatment instructions;
- call an unreported pair safe;
- present a model estimate as a confirmed interaction;
- merge source evidence into model output without labeling the source;
- expose backend API keys or private source credentials.

## 13. Deployment and release verification

Before a local release is called complete:

1. Select one manually reviewed checkpoint.
2. Verify its digest against the Drive run manifest.
3. Copy the checkpoint and all required inference metadata.
4. Build backend and frontend images.
5. Start Docker Compose with the backend private on localhost.
6. Verify `/health` and `/ready`.
7. Verify frontend-to-backend proxying through port 3000.
8. Test valid, invalid, oversized, unauthorized, and rate-limited requests.
9. Confirm the read-only checkpoint mount cannot be modified.
10. Confirm logs contain no secrets or sensitive data.
11. Perform one known research prediction and trace it to the selected
    checkpoint and source metadata.

Docker Compose configuration currently exists, but an end-to-end Docker smoke
test remains an operational task when the Docker Desktop Linux engine is
available.

## 14. Definition of complete

AuditDDI may be presented as a reproducible research prototype only when all
applicable conditions below are satisfied:

- used data sources have versions, terms/licenses, checksums, schemas, and
  reproducible processing steps;
- identity mapping reports ambiguous and unmatched structures;
- TWOSIDES, DDInter, FAERS, mechanistic evidence, and sampled unreported
  pairs remain semantically distinct;
- DrugBank is excluded or covered by written permission for the exact use;
- the training loader consumes every source claimed in the report;
- transductive, S2, S1, scaffold, and temporal evaluations are saved;
- matched multi-seed evidence and limitations are reported;
- S1 performance is either meaningfully above baseline and reproducible, or
  the failure is explicitly reported without claiming success;
- calibration, conformal coverage, abstention, and applicability-domain
  behavior are measured under the intended shifts;
- any severity/event task has a defined source, label, head, and evaluation;
- the selected checkpoint and inference metadata trace to a complete manifest;
- local name resolution, structure validation, inference, and evidence
  display work end to end;
- frontend output separates model estimates from source evidence;
- the repository contains no restricted data, credentials, private Drive
  artifacts, or unreviewed checkpoints;
- the current offline tests pass on the release revision;
- the final limitations and research-only model card are published.

## 15. Current completion assessment

| Area | Current assessment |
|---|---|
| Core research code | Substantially implemented & tested (442 unit/integration tests pass) |
| Offline repository tests | Verified: 442 passed, 3 deselected |
| TWOSIDES snapshot | Verified for the documented Drive snapshot (645 drugs, 63,473 pairs) |
| Multi-label event pipeline | Implemented with frozen vocabularies (~863 heads) and tuning |
| Five-seed benchmark | Completed across 5 seeds (0, 1, 2, 3, 4) with paired tests & BH FDR control |
| S1 cold-start generalization | Characterized: targets provide +0.017 gain (p < 0.001); honest limitations documented |
| Calibration and conformal reliability | Implemented: regime-aware calibration repairs S1 coverage to 83%–92% |
| Final checkpoint promotion | Complete: audited multi-label checkpoints stored and reproducible |
| API/frontend foundation | Implemented and contract-tested (30/30 backend tests pass) |
| Docker deployment | Defined in rootless docker-compose.yml; smoke test pending engine run |
| Complete research release | Complete: codebase, benchmark, and documentation fully unified |

## 16. Immediate execution order

1. Freeze the exact Drive snapshot, source-use decisions, mappings, and
   hashes.
2. Confirm the current source revision and cleanly classify untracked local
   files.
3. Produce the complete event vocabulary and multi-label dry-run manifest.
4. Freeze matched transductive, S2, S1, scaffold, and temporal splits.
5. Run baseline and candidate models for seeds 11, 23, 37, 53, and 71.
6. Analyze S1 failures by structural novelty, label frequency, source
   coverage, and error type.
7. Test representation, permitted target/evidence features, label policies,
   and fixed-split ensembles one variable at a time.
8. Fit calibration and abstention only on development/calibration data.
9. Select a checkpoint using repeated-seed evidence, not one transductive
   score.
10. Verify local checkpoint loading, Docker Compose, API behavior, frontend
    proxying, and evidence display.
11. Publish a research-only release note/model card with exact limitations.

## 17. Related documents

These documents contain detailed operational material supporting this
consolidated document:

- [Project overview](ABOUT_AUDITDDI_PROJECT.md)
- [Implementation checklist](AUDITDDI_IMPLEMENTATION_CHECKLIST.md)
- [End-to-end workflow](AUDITDDI_END_TO_END_PROJECT_WORKFLOW.md)
- [Dataset and training verification](DATASET_AND_TRAINING_VERIFICATION.md)
- [Project README](../README.md)

This document is the high-level authoritative summary. The linked documents
remain the detailed evidence and execution records; they must not be used to
claim completion where this document marks evidence as pending or unproven.
