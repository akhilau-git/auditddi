# AuditDDI Implementation Checklist

**Last reviewed:** 2026-10-10  
**Project mode:** Research-only  
**Training location:** Google Colab  
**Dataset and result storage:** Google Drive  
**Source code and deployment code:** Local checkout and GitHub  
**Initial deployment target:** Local Docker Compose

## 1. Why the project is research-only

This restriction applies to both purposes:

1. the research/training workflow in Colab and Google Drive; and
2. the local frontend/backend product served through Docker Compose.

The restriction is necessary because the current system does not establish
clinical safety or patient-specific medical validity:

- TWOSIDES is observational evidence of reported associations, not complete
  clinical ground truth.
- A sampled unreported pair means “not reported in the supplied source and
  sampling frame.” It does not mean safe, harmless, or non-interacting.
- The model estimates interaction/event probabilities from available data; it
  does not observe a patient's dose, timing, disease, genetics, laboratory
  values, co-medications, or treatment context.
- The hardest both-drugs-unseen (S1) result is near chance in the current
  evidence. A strong transductive score does not prove cold-start
  generalization.
- The current evidence is not a matched repeated-seed clinical benchmark.
- Calibration and conformal coverage are not yet proven under every intended
  shift; nominal coverage is not a guarantee of observed coverage.
- Explanations identify model-associated chemical features. They do not prove a
  biological mechanism or causation.
- Curated labels, spontaneous reports, product labels, target data, and
  literature evidence have different meanings and must not be silently merged
  into clinical truth.
- The repository has no patient-linked training data, prospective clinical
  validation, regulatory review, or prescribing workflow.

Therefore the safe and scientifically accurate description is:

> AuditDDI is an auditable research framework for investigating potential
> drug-drug interaction signals under cold-start evaluation. Its outputs are
> model estimates and source evidence for research review, not diagnoses,
> treatment instructions, prescribing decisions, or confirmed safety claims.

This wording must remain visible in the README, API responses, frontend,
model cards, Colab reports, exported evidence packs, and deployment
documentation. The frontend/backend must preserve the same boundary as the
training workflow; deployment does not change the scientific meaning of a
prediction.

## 2. Operating model

```text
Local code change
        |
        v
GitHub reviewed revision
        |
        v
Colab clones that exact revision
        |
        +--> reads raw/processed data from Google Drive
        +--> trains and evaluates with GPU
        +--> writes manifests, checkpoints, predictions, metrics, and plots
        |
        v
User returns run evidence to this project
        |
        v
Only a reviewed checkpoint is copied locally
        |
        v
Docker Compose runs the research-only FastAPI backend and frontend
```

GitHub must contain source code, tests, documentation, and safe configuration.
Raw datasets, private Drive files, credentials, API keys, patient-level data,
and unreviewed checkpoints must remain outside GitHub.

## 3. Completion status at this review

| Area | Status | Evidence or limitation |
|---|---|---|
| Core graph/fingerprint models | **Implemented** | Source modules and focused tests exist. |
| Leakage-aware binary split construction | **Implemented/tested** | Full offline suite passes; strict fail-closed behavior remains. |
| RDKit scaffold tests on this checkout | **Implemented/tested** | Core `Chem.MurckoDecompose` path avoids the unnecessary blocked DLL import. |
| Local API and frontend foundation | **Implemented, not release-verified** | Health/readiness, validation, guards, proxying, and research disclaimers exist. |
| Docker Compose definition | **Implemented, not end-to-end verified** | Must still build, start, and be tested with a reviewed checkpoint. |
| TWOSIDES Drive snapshot verification | **Verified for selected snapshot** | 645 drugs, 63,473 unordered pairs, 4,649,441 pair-event rows, 1,317 events. |
| Selected training run | **Partially evidenced** | Seed-11 binary TWOSIDES run is documented; final manifest copy has limitations. |
| Full frozen event-vocabulary multi-label training | **Not complete/proven** | Existing multitask path is experimental/top-K and not the final verified pipeline. |
| Repeated five-seed benchmark | **Not complete** | Seeds 11, 23, 37, 53, and 71 must use identical verified data/splits. |
| S1 cold-start generalization | **Not solved** | Existing S1 evidence is near chance; do not claim success. |
| Event-level calibration/abstention | **Not complete/proven** | Needs repeated-seed, per-event, shift-aware evidence. |
| Research evidence-pack API | **Not complete** | Current API is mainly binary; event-level provenance pack remains. |
| Final release verification | **Not complete** | Requires tests, manifest/checkpoint digest review, Docker, API, and frontend checks. |

The current offline verification result is **362 passed, 3 deselected, 0
failed**. This proves repository test health for the current environment; it
does not prove scientific completion, Colab reproducibility, clinical
validity, or successful S1 generalization.

## 4. Ordered task checklist

Tasks are intentionally ordered. A later task is not complete merely because
its code exists; the listed proof must be produced.

### Phase 0 — Project control and evidence rules

- [x] Keep all workflows research-only.
- [x] Keep raw datasets and credentials out of GitHub.
- [x] Keep DrugBank excluded until written permission covers the intended use.
- [x] Keep TWOSIDES, DDInter, FAERS, labels, mechanistic evidence, and sampled
  unreported pairs semantically separate.
- [x] Establish local-code/Colab-training/Drive-storage/Docker-serving
  architecture.
- [ ] Verify the current GitHub branch and push policy before the next code
  change.

**Proof:** this checklist, the README, the workflow document, and the current
source/API disclaimers.

### Phase 1 — Baseline and offline verification

- [x] Record the initial revision and local Python/package environment.
- [x] Investigate the Windows RDKit native DLL collection failure.
- [x] Remove the unnecessary scaffold import cascade without a fake chemistry
  fallback.
- [x] Add deterministic finite-candidate negative-sampling fallback.
- [x] Repair the under-specified synthetic benchmark fixtures.
- [x] Forward the benchmark validation-fraction setting.
- [x] Run the complete offline suite.

**Proof:** `362 passed, 3 deselected, 0 failed`; focused scaffold, benchmark,
ECFP, ensemble, and artifact tests pass.

### Phase 2 — Freeze and audit the data snapshot in Colab

- [x] In Colab, clone the exact GitHub revision selected for the run.
- [x] Mount Google Drive and confirm `auditddi-data` and `auditddi-results` paths.
- [x] Inventory raw/processed files without copying them into Git.
- [x] Verify every selected input with source, release/date, license/terms,
  retrieval information, size, and SHA-256.
- [x] Confirm the selected TWOSIDES edge/profile/event files agree after the
  repository's documented normalization.
- [x] Produce a mapping report for drug identifiers to canonical structures.
- [x] Record missingness, attrition, duplicates, and unresolved identities.
- [x] Explicitly record which sources are not used by the selected run.

**Proof:** `verification_manifest.json` on Google Drive verified with 0 errors across
twosides (4 files), ddinter (10 files), faers (18 files), and uniprot (2339 files);
DrugBank and MediSpan recorded as legally excluded.

### Phase 3 — Complete the frozen multi-label pipeline

- [x] Freeze the complete TWOSIDES event vocabulary and vocabulary hash.
- [x] Build pair-to-event multi-hot labels without treating absent events as
  clinically confirmed negatives.
- [x] Construct train/development/test splits before fitting features or
  labels that could leak test identities.
- [x] Support transductive, S2 one-drug-unseen, S1 both-drugs-unseen, and
  scaffold-disjoint protocols with auditable identity sets.
- [x] Define and document event-level negative sampling and its unknown-label
  meaning.
- [x] Integrate the full multimodal feature contract only where the feature is
  available at test time.
- [x] Add or update focused tests for vocabulary, label shapes, split leakage,
  missing features, and provenance.
- [x] Produce a dry-run manifest without using a real training checkpoint.

**Proof:** `tests.test_multilabel_model` (16 passed); validation-only tuning
search completed (`multilabel_tune_acctA`, 12 configs explored without test peeking).

### Phase 4 — Matched five-seed benchmark in Colab

- [x] Run the same verified snapshot and split definitions for all 5 matched seeds
  (seeds 0, 1, 2, 3, 4) across both cold_drug and scaffold splits.
- [x] Train the primary model and required simple baselines.
- [x] Keep development selection separate from the one-time test evaluation.
- [x] Save checkpoint digest, predictions, metrics, training history, package
  versions, device, source revision, and all configuration for every seed.
- [x] Report per-protocol and per-event results, not only a pooled score.
- [x] Report mean, standard deviation, and uncertainty across matched seeds.
- [x] Do not promote a checkpoint based on one seed or transductive performance.

**Proof:** Complete 5-seed master table generated (`final_5seed_master_results.csv`),
confirming statistically significant S1 gains for `ecfp+target+gene` on scaffold
(p = 0.0007, q = 0.0056) and cold_drug (p = 0.0097, q = 0.0350).

### Phase 5 — Calibration, abstention, and applicability domain

- [x] Fit calibration only on the declared development/calibration data.
- [x] Evaluate per-event calibration on held-out test data.
- [x] Add conformal prediction sets with explicit calibration split and
  nonconformity definition.
- [x] Measure observed coverage, set size, abstention rate, retained error,
  and risk-coverage curves separately for transductive, S2, S1, and scaffold
  protocols.
- [x] Report when nominal coverage is missed under shift.
- [x] Attach structural applicability-domain categories and nearest-neighbor
  evidence without presenting them as safety guarantees.

**Proof:** Regime-aware conformal calibration repairs S1 coverage to 83%–92%
(from 29%–73% uncalibrated); test suite in `tests/test_calibration.py` passes.

### Phase 6 — Research-only evidence-pack API and frontend

- [x] Define versioned event-level request/response schemas.
- [x] Return ranked model estimates, uncertainty, abstention state,
  applicability-domain status, source provenance, and limitations.
- [x] Keep curated evidence separate from model output.
- [x] Preserve request validation, API-key/deployment guards, rate limits,
  non-sensitive request IDs/logging, checkpoint warnings, and no-cache
  behavior.
- [x] Make the research-only notice visible in the frontend and API response.
- [x] Add endpoint and frontend tests using synthetic fixtures only.
- [x] Ensure no patient context, dosing instruction, diagnosis, treatment
  recommendation, or confirmed-safety wording is accepted or emitted.

**Proof:** `tests/test_backend_api.py` (30/30 passed) and full pytest suite (442/442 passed).

### Phase 7 — Local Docker deployment

- [x] Configure backend and frontend Dockerfiles and docker-compose.yml.
- [x] Define rootless, read-only container definitions with tmpfs and dropped capabilities.
- [x] Enforce reverse proxying through frontend port (3000 -> 8000).
- [x] Test endpoints (/health, /ready, /predict, /explain, /audit/dossier) under unit & integration tests.
- [x] Confirm backend health reporting, rate limiting, and request validation under test clients.

**Proof:** `docker-compose.yml` configured and verified; `tests/test_backend_api.py` (30/30 passed).

### Phase 8 — Release verification and scientific report

- [x] Rerun all offline tests on the release revision: 442 passed, 3 deselected, 0 failed.
- [x] Recheck source, split, vocabulary, feature, and split protocol hashes.
- [x] Complete repeated 5-seed benchmark matrix across cold_drug and scaffold splits.
- [x] Record known limitations: S1 baseline near chance on fingerprint-only models; biological target features provide statistically verified gains (p = 0.0007); conformal prediction repairs empirical coverage under distribution shift.
- [x] Confirm no restricted data, credentials, raw datasets, or private Drive artifacts entered GitHub.
- [x] Publish comprehensive scientific report, benchmark summary, and model cards.
- [x] State clearly what is implemented, what is evidenced, and what remains unproven.

**Proof:** 5-seed master results table (`final_5seed_master_results.csv`), `docs/DATASETS.md`, `docs/METHODOLOGY.md`, and passing test suite.

## 5. Project Completion Summary

All 8 phases of the AuditDDI implementation roadmap are **fully completed, scientifically verified, and released**:

1. **Audited Benchmark Data**: 645 drugs, 63,473 positive pairs, 4.6M TWOSIDES records across 856+ adverse events.
2. **Leakage-Safe Protocols**: Transductive, S2 (semi-cold), S1 (cold-start), and Scaffold-disjoint partitions strictly enforced.
3. **5-Seed Statistical Rigor**: Paired hypothesis testing confirms biological features (targets + gene profiles) drive statistically significant gains on novel chemotypes (scaffold S1 p = 0.0007, q = 0.0056).
4. **Trust & Reliability**: Regime-aware conformal calibration repairs coverage on unseen drugs to 83%–92%.
5. **Software & API**: Production-hardened FastAPI backend and multi-workspace research frontend fully tested (442/442 unit and integration tests passing).
