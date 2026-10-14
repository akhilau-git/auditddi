# AuditDDI: end-to-end project workflow

This document is the working plan for completing **AuditDDI: An Auditable Graph-Fingerprint Framework for Cold-Start Drug-Drug Interaction Prediction**. It explains what belongs in Google Drive, what runs in Colab, what runs on the local computer, and the checks that must pass before calling the project complete.

## Important boundaries before starting

1. **Training runs in Google Colab.** Raw and processed datasets, run manifests, and training outputs belong in the Google Drive folder `auditddi-data` (or the results folder described below).
2. **The application runs locally.** The trained model is copied from Drive to the local backend checkpoint folder when a run is selected for the app.
3. **Never put data or credentials in GitHub.** Do not commit `.env`, API tokens/passwords, downloaded databases, API response caches, model checkpoints, patient-level data, or private Drive files.
4. **DrugBank is gated by written permission and an account that can download the data.** You said you have emailed DrugBank. Until they confirm the specific intended uses in writing, do not bulk-download, retain, train on, benchmark with, or display DrugBank data. The current release page may also show downloads as unavailable. An API key or portal login by itself does not establish permission for every use.
5. **Do not treat missing evidence as a safe pair.** TWOSIDES and DDInter have different collection and label meanings. A pair not listed in one source is normally *unknown*, not a confirmed non-interaction.
6. **This is a research prototype.** Do not present model output as a clinical decision, diagnosis, treatment recommendation, or validated severity assessment.

## What the current project does and does not establish

- The existing model pipeline is primarily trained from TWOSIDES-derived interaction labels. The DDInter importer prepares DDInter data; importing files does not automatically make the training pipeline learn from them.
- The current prediction interface is a binary interaction signal and requires molecular structures (SMILES). It does not currently establish a calibrated severity prediction. Severity needs trustworthy severity labels, a separately defined task/head, appropriate evaluation, and clear source provenance.
- The existing integration code has several source-specific data-preparation paths, but that is not proof that every listed source is linked to the same drugs or used by the model. Audit each importer, identity mapping, feature builder, and training loader before claiming an integration.
- Prior project notes report a strong transductive score but near-chance performance for the hardest both-drugs-unseen split. Re-run the evaluation from saved predictions and manifests before making performance claims. A high random-split score alone does not show cold-start capability.
- A DrugBank API result, DrugBank release download, DDInter record, FDA label, and spontaneous adverse-event report are different kinds of evidence. Keep them separate and label each one.

## The whole workflow at a glance

```text
GitHub source code ──clone──> Colab notebook/runtime
                                  │
Google Drive/auditddi-data <──────┤ raw files, normalized tables, manifests
                                  │
                                  └──> identity resolution + evidence graph
                                            │
                                            └──> training + cold-start evaluation
                                                      │
Google Drive/auditddi-results <── checkpoints, metrics, logs, run manifest
         │
         └── selected checkpoint copied to local backend/checkpoints
                    │
                    └── local FastAPI backend <──> local web dashboard
```

## Phase 0 — Make a safe project copy

### On your computer

1. Open PowerShell and go to the repository:

   ```powershell
   cd "D:\Drug-Drug Interaction\AuditDDI"
   git status --short --branch
   git remote -v
   ```

2. Confirm the remote is your intended repository: [akhilau-git/auditddi](https://github.com/akhilau-git/auditddi).
3. Keep a backup of important local changes before switching branches or pulling. Do not use `git add .` until you have checked every changed and untracked file.
4. Read the root README and this workflow. Resolve conflicting instructions in the code/notebooks before training.

### In Google Colab

Use a GPU runtime when training. Clone the repository into the temporary Colab VM; the VM can be deleted at any time, so it is not the storage location for data or final model artifacts.

```python
%cd /content
!git clone https://github.com/akhilau-git/auditddi.git AuditDDI
%cd /content/AuditDDI
!git status --short --branch
```

Record the checked-out commit in every experiment manifest. If you need local unpushed changes in Colab, push reviewed source changes to GitHub first, then clone that commit. Do not copy datasets into the repository.

## Phase 1 — Connect Colab to Google Drive and inventory the files

1. In Colab, mount Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Set paths. Change `MyDrive` only if your Drive layout is different:

   ```python
   from pathlib import Path

   DATA_ROOT = Path('/content/drive/MyDrive/auditddi-data')
   RESULTS_ROOT = Path('/content/drive/MyDrive/auditddi-results')
   DATA_ROOT.mkdir(parents=True, exist_ok=True)
   RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
   print('Data:', DATA_ROOT, DATA_ROOT.exists())
   print('Results:', RESULTS_ROOT, RESULTS_ROOT.exists())
   ```

3. Inventory before processing. List folder names, files, sizes, and available README/license files. Do not upload the inventory if it exposes private information. Confirm the existing datasets and notebooks are present before running a downloader or notebook.
4. Run the repository's Colab census helper if present:

   ```python
   %run /content/AuditDDI/scripts/colab_drive_census.py
   ```

   If this path is absent in the GitHub revision you cloned, use a simple file listing and report the folder names, file names, and file sizes. Do not share access links or credentials.
5. Preserve all original files as **raw/immutable** inputs. Make normalized outputs in a separate `processed` folder. Never overwrite the original downloaded dataset.

Recommended Drive organization (keep actual vendor filenames and release folders under `raw`):

```text
MyDrive/
├── auditddi-data/
│   ├── TWOSIDES/raw/ and processed/
│   ├── DDInter/raw/ and processed/
│   ├── DrugBank/raw/ and processed/       # only after written permission
│   ├── PubChem/raw/ and processed/
│   ├── ChEMBL/raw/ and processed/
│   ├── OpenFDA/raw/ and processed/
│   ├── PharmGKB/raw/ and processed/
│   ├── FAERS/raw/ and processed/
│   ├── BindingDB/raw/ and processed/
│   ├── UniProt/raw/ and processed/
│   ├── PDB/raw/ and processed/
│   └── GEO/raw/ and processed/             # if actually used
└── auditddi-results/
    └── runs/<run-id>/  # manifest, logs, metrics, predictions, checkpoints
```

For every file, record source, release/date, download date, license/terms URL, source URL or query, SHA-256, and processing script version in a manifest. Never place DrugBank credentials or a private download URL in the manifest.

## Phase 2 — Confirm what each source is allowed to say

| Source | Appropriate role in this project | Do not silently interpret it as |
|---|---|---|
| TWOSIDES | Observational DDI signal labels; current baseline supervision | Complete clinical truth; absent pair means safe |
| DDInter | Curated DDI records, often with severity/mechanism fields; complementary labels and possible source-specific task | Exhaustive list; absent pair means no interaction |
| DrugBank | Licensed reference/database or API evidence only for uses DrugBank confirms | Freely redistributable data or automatically permitted model training/benchmarking |
| PubChem | Compound identity, CID and structure resolution (including SMILES where available) | Clinical interaction evidence |
| ChEMBL | Compound, assay, target and bioactivity evidence | DDI labels |
| BindingDB | Drug/compound–target affinity evidence | DDI labels |
| PharmGKB | Pharmacogenomic drug–gene/variant evidence, subject to its terms | A general pairwise DDI database or assumed public live API |
| UniProt | Protein identity, sequences and target annotations | Drug interaction labels |
| PDB | Experimental 3D structure evidence | Complete drug-target coverage |
| FAERS | Spontaneous adverse-event reports and derived signals | Causal proof or confirmed DDI labels |
| OpenFDA | Public label and safety-report fields; evidence extraction with citations | A definitive, normalized pairwise interaction truth set |
| GEO | Relevant expression/omics evidence where a defensible drug/target mapping exists | DDI labels or a mandatory feature source |

Use only the sources that are actually present, accessible, licensed for the planned use, and linked with auditable identity mappings. More datasets do not automatically mean a better model.

## Phase 3 — Prepare DDInter (can proceed under its published terms)

1. Review [DDInter's download page](https://ddinter2.scbdd.com/download/) and [terms](https://ddinter2.scbdd.com/terms/) before redistribution or publication. DDInter lists downloadable category files; it is not necessarily the entire database.
2. Download the permitted category CSV files into `auditddi-data/DDInter/raw/<release-or-download-date>/`. Preserve their original names.
3. Use the repository importer `src/data_prep/ddinter_pipeline.py` only after checking its input paths, expected headers, license notes, and output paths against the files you actually downloaded. Do not assume an importer has trained the model.
4. Save normalized output and a row-count/schema report under `DDInter/processed/`. Review a sample of rows, missing IDs, duplicate pairs, severity vocabulary, and drug identifier types.
5. Keep DDInter positives and any severity/mechanism labels in a source-specific table. Missing DDInter pairs stay unknown. Do not label all unlisted pairs as negatives.

## Phase 4 — DrugBank download, only after authorization

The official [DrugBank download help](https://portal.drugbank.com/releases/help) describes authenticated download requests and release metadata. The [release page](https://go.drugbank.com/releases/latest) determines which files are actually available to your account at that time. The URL in the shell example from the help page is a download route, not proof that your account has access or that every use is permitted. The release page can temporarily mark downloads unavailable.

**Wait for DrugBank's written response** to confirm the exact intended uses: academic research, model training, external benchmarking, retaining files/results in Google Drive, and displaying results in a student prototype. Ask about rate limits, retention, citations, redistribution, and whether the API and full release have different terms. If permission is limited, follow the limit or omit DrugBank from that task.

Once your permission and account access are confirmed:

1. Use the authenticated portal/release metadata to discover the exact filenames and direct download links available to your account. Do not invent or guess a ZIP URL from a `go.drugbank.com` example.
2. Keep the login in Colab Secrets (or another secret manager); never paste it into a notebook cell, shell history, source file, GitHub, or a shared notebook. Use an authenticated request that reads secrets at runtime. Do not print the secret or the authenticated URL.
3. Download only the release/files and fields allowed by the written permission. Check the HTTP status, expected file type, and ZIP integrity before treating a response as a dataset; an HTML login/error page can otherwise be saved with a `.zip` name.
4. Save under `auditddi-data/DrugBank/raw/<release>/`, with a private access-controlled Drive folder. Record version, checksum, retrieval date, and permission/retention reference without recording credentials.
5. Parse only required fields and only for authorized uses. Keep raw licensed files private. Do not push them, derived redistribution files, or API response caches to GitHub.
6. The existing `src/data_prep/collect_drugbank_data.py` is not an approved turnkey bulk downloader. Do not run it until its endpoint, identifier mapping, permission, and output handling are checked against official DrugBank documentation and your written authorization.

The repository includes `scripts/download_drugbank_release.py` for the documented full-database route. It first reads the authenticated release list, confirms the requested version appears there, follows redirects, streams the archive to a temporary file, verifies ZIP integrity, and writes a checksum manifest. It does not bypass account access or licensing. In Colab, add `DRUGBANK_EMAIL` and `DRUGBANK_PASSWORD` as Colab Secrets, allow notebook access to them, then run:

```python
from google.colab import userdata
import os

os.environ['AUDITDDI_DRUGBANK_EMAIL'] = userdata.get('DRUGBANK_EMAIL')
os.environ['AUDITDDI_DRUGBANK_PASSWORD'] = userdata.get('DRUGBANK_PASSWORD')
```

```python
%cd /content/AuditDDI
!python scripts/download_drugbank_release.py --version 5.1.22
```

The default destination is `/content/drive/MyDrive/auditddi-data/DrugBank/raw/5.1.22/`. For a different version, pass its dotted version, such as `--version 5.1.21`; `--version latest` selects the first version in DrugBank's authenticated release list. Do not use `--force` unless you intentionally want to replace an existing archive. If DrugBank returns an access error or non-ZIP response, stop and contact DrugBank rather than trying guessed URLs or alternate credentials. Delete the runtime environment variables when finished and never print them.

If DrugBank does not permit training/benchmarking or retention, you can still complete a research pipeline using permitted sources; leave DrugBank out of those stages and label the limitation honestly.

## Phase 5 — Use public/approved API sources carefully

For each API, read its current documentation and terms, identify rate limits and caching/retention rules, and store only fields needed for the project.

- **PubChem PUG REST:** use a drug name only to find candidate compounds; resolve ambiguity with CID/InChIKey and a reviewed crosswalk. Obtain canonical/isomeric SMILES from the selected compound when available. Names can map to multiple salts, stereoisomers, or formulations. Never accept the first fuzzy name match silently.
- **ChEMBL:** use official REST resources for compound/target/assay evidence where useful. Preserve assay type, organism, units, relation operator, and source identifiers; do not collapse unlike assays into one number.
- **OpenFDA:** retrieve relevant label fields with source and update metadata. Extract interaction statements as cited evidence that may need manual/curated normalization. A label hit is not a universal pairwise ground-truth label.
- **PharmGKB:** use the files and access method allowed by its current terms. Do not assume there is an unauthenticated live API suitable for polling or bulk downloading.
- **FAERS:** use official releases or permitted endpoints. Retain reporting-period/source fields; deduplicate reports as appropriate and describe signals as reports, not causal evidence.
- **BindingDB, UniProt, PDB, GEO:** use approved release/API access and source IDs. Add these only when a reproducible mapping to the drug or target is available and the information is relevant to the model.

Use an explicit user selection when compound search returns multiple candidates. Cache API responses in Drive only when the source terms allow it. Use retries/backoff and respect published request limits.

## Phase 6 — Build the cross-source identity map and evidence graph

Do not merge all CSVs by drug name into one giant flat spreadsheet. Names are not stable identifiers, and source labels do not all mean the same thing.

Create at least these logical records (the exact table/Parquet format can follow existing project conventions):

1. `drug_entity`: stable internal drug ID, preferred name, salt/form/parent relationship, and status.
2. `drug_identifier`: internal ID, source name, source ID (for example PubChem CID, DrugBank ID if permitted, DDInter ID), identifier type, and mapping method.
3. `structure`: internal ID, SMILES/InChI/InChIKey, standardization method/version, source, and validation status.
4. `interaction_evidence`: canonical unordered pair, source, source record ID, label type, value, severity/mechanism if supplied, evidence text/reference, collection date, and confidence/curation status.
5. `target_evidence`: drug–protein/target relation, source, assay/affinity details, organism, and provenance.
6. `pharmacogenomic_evidence`: drug–gene/variant relation and source-specific annotations.
7. `report_signal` and `label_evidence`: FAERS/OpenFDA evidence kept distinct from curated DDI labels.
8. `mapping_review`: unresolved/ambiguous mappings, reviewer decision, and reason.

Canonicalize unordered drug pairs consistently (`min(drug_id), max(drug_id)`) but preserve source pair IDs. Preserve conflicts: if two sources disagree, keep both evidence records and expose the disagreement. Use deterministic structure standardization, preserve the original structure, and log salt/stereo decisions. Make a mapping report: matched, unmatched, ambiguous, invalid structure, duplicate, and excluded counts for each source.

## Phase 7 — Connect prepared evidence to training without label leakage

1. Start with a reproducible TWOSIDES baseline matching the existing model path. Record the exact source release, transformation, negative-sampling method, features, and splits.
2. Add DDInter as a separate source-specific supervision task only after schema and mapping review. If merging labels for a shared task, define the label semantics and conflict policy first. Do not count DDInter as independent test evidence if its labels were used for training or model selection.
3. Keep DrugBank out until written permission covers the intended training/evaluation/retention. If approved, treat its labels/evidence as a separate source and document any licensing restriction on results.
4. Use ChEMBL/BindingDB/PharmGKB/UniProt/PDB/GEO as typed auxiliary features or auxiliary prediction tasks only when mapping and data quality support it. They do not become positive/negative DDI labels just because they share a drug ID.
5. Keep FAERS/OpenFDA as distinct evidence or carefully defined auxiliary tasks. Avoid turning mentions/report co-occurrences into causal interaction truth without a validated curation process.
6. Use source-specific loss masks/heads when label meanings differ. A model can share graph representations while learning only from labels that exist for each row/task.
7. Establish splits before feature fitting. Keep the same canonical pair out of train/validation/test across every source to avoid cross-source leakage. Fit vocabularies, scalers, graph propagation, sampling, and calibration on training data only.
8. Store training config, source manifest, code commit, dependency versions, random seeds, split IDs, class counts, and output checksums for every run.

## Phase 8 — Train and evaluate in Colab (GPU)

### Before training

- Select a Colab GPU runtime and record the GPU/runtime details.
- Install from the repository's pinned Colab requirements, checking compatibility with the notebook first:

  ```python
  %cd /content/AuditDDI
  !pip install -r requirements_colab.txt
  ```

- Confirm the Drive paths and source manifests are correct. Print file counts and schemas, not secrets or private records.
- Run the project's training entrypoint/notebook from a clean runtime and save logs to the run's Drive folder.

### Required evaluation views

Evaluate more than a random pair split:

1. **Transductive pair split:** both drugs may have appeared in training; useful baseline only.
2. **S1 cold start:** both drugs are unseen during training.
3. **S2 cold start:** one drug is unseen; report both directions if the protocol distinguishes them.
4. **Scaffold split:** hold out chemical scaffolds to test chemical generalization.
5. **Temporal split:** where source dates allow, train on earlier evidence and evaluate on later evidence.

Use multiple seeds and report mean, spread, sample counts, class balance, and confidence intervals where feasible. For binary tasks report AUROC and especially AUPRC, plus threshold-specific precision/recall/F1 and calibration. For severity use an explicitly ordered severity label and suitable ordinal metrics; report per-class support and weighted agreement/error. Report each source/task separately. Do not say “95% severe” unless a calibrated severity head was trained and validated for that meaning.

**Completion gate:** the both-unseen result must be meaningfully above the appropriate baseline on a leakage-safe split, reproducible across seeds, and accompanied by error analysis. If it is not, report that limitation and keep improving the method. Do not select a model using the final test set.

## Phase 9 — Promote a checkpoint to the local app

1. In Drive, select one run only after comparing it with the documented baseline and checking its source permissions and evaluation.
2. Keep the run checkpoint, metrics, manifest, and model/config version together in `auditddi-results/runs/<run-id>/`.
3. Download/copy the selected checkpoint to the local repository's `backend/checkpoints/` using the filename and model architecture expected by the backend. The project `.gitignore` excludes checkpoints; keep it that way.
4. Copy any required vocabulary, feature schema, structure normalization settings, calibrator, and model metadata too. A weight file alone may not be enough to reproduce inference.
5. Confirm the backend loads the selected checkpoint and reports the correct model/source version. Do not overwrite a working model until the candidate has been reviewed.

## Phase 10 — Connect the local backend, source lookups, and frontend

The model needs a structure, while a user's typed drug name is only a name. Implement the flow as separate auditable steps:

1. **Search:** user types a name; backend returns candidate compounds and identifiers from permitted sources.
2. **Resolve:** user selects the exact compound/form; backend records CID/InChIKey and obtains/validates SMILES (PubChem is one possible approved resolver). Display the selected form so the user can catch the wrong salt or stereoisomer.
3. **Predict:** pass validated structures and model-required features to the local model. Return the research prediction, model version, uncertainty/applicability information, and known limitations.
4. **Retrieve evidence separately:** permitted DDInter/OpenFDA/DrugBank evidence is fetched as source evidence, not silently substituted for the model prediction. DrugBank calls are disabled until authorization and secrets are configured.
5. **Audit display:** show source name, record/link, retrieval date, label meaning, and conflicts. Distinguish “model estimate” from “database/label evidence.” Do not invent official confirmation or severity.

Keep API credentials on the backend in environment variables or a secret manager; never put them in browser JavaScript. Add timeouts, input validation, rate limits, sanitized errors, and an offline/model-unavailable state. Do not claim the frontend is fully connected until a real local request completes and the response shown is traceable to the loaded checkpoint and selected drug identities.

## Phase 11 — Run the app locally

From PowerShell in the repository, create/use a local environment and install the project dependencies according to the README. Never paste a password into a command that will be saved in shell history.

For a direct backend run, from the repository root:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

If Docker Compose is the chosen route, review the Compose configuration and then start the app:

```powershell
docker compose up --build
```

Use the frontend address documented by the current Compose/README configuration (the project has used `http://localhost:3000`) and the backend at `http://127.0.0.1:8000`. Verify the health/ready endpoint documented by the API, confirm the logs show the expected checkpoint, resolve a known drug pair, and make one research prediction. Confirm the page displays source provenance and uncertainty. Keep the app local; do not expose it publicly until security, licensing, and validation are reviewed.

## Phase 12 — Keep GitHub clean and reproducible

Before committing, inspect every file:

```powershell
git status --short
git diff --check
git diff
```

Stage named source/documentation files, not the whole working tree. Example:

```powershell
git add docs/AUDITDDI_END_TO_END_PROJECT_WORKFLOW.md
git add path/to/reviewed/source_file.py
git status --short
git diff --cached --stat
git diff --cached
```

Then commit and push only after confirming the staged list contains no data, `.env`, API keys, DrugBank material, generated results, or checkpoints:

```powershell
git commit -m "Document AuditDDI end-to-end workflow"
git push origin main
```

If a secret was ever committed, removing it in a later commit is not sufficient; revoke/rotate it and follow the repository's history-cleanup process.

## Definition of “complete”

The project is ready to present as a reproducible research prototype only when all applicable items below are checked:

- [ ] Google Drive inventory is known; each used source has a version, terms/license record, checksum, schema report, and reproducible raw-to-processed step.
- [ ] Drug identity/structure mapping has reviewed ambiguous cases and reports unmatched cases.
- [ ] TWOSIDES and DDInter labels remain semantically distinct; missing records are not silently treated as negatives.
- [ ] Any DrugBank use is covered by written permission for the exact use and retention; otherwise the pipeline works without it.
- [ ] Training code actually consumes the sources claimed in the report; importing a file alone is not described as training integration.
- [ ] Leakage-safe transductive, one-unseen, both-unseen, and scaffold/temporal evaluations are saved with the run; multi-seed results and limitations are reported.
- [ ] Any severity prediction has a defined source/label/task and its own evaluation; otherwise the app labels output as binary interaction research signal only.
- [ ] The selected checkpoint and all inference metadata load locally and are traceable to a Drive run manifest.
- [ ] Drug name resolution, exact compound selection, SMILES generation/validation, model inference, and evidence display work end-to-end locally.
- [ ] Frontend clearly separates model estimates from retrieved source evidence and displays uncertainty and provenance.
- [ ] GitHub contains reviewed code/docs only; no datasets, licensed DrugBank files/results, credentials, or checkpoints.

## Official source links

- [DrugBank API documentation](https://docs.drugbank.com/discovery/v1/)
- [DrugBank release download help](https://portal.drugbank.com/releases/help)
- [DrugBank latest release page](https://go.drugbank.com/releases/latest)
- [DDInter downloads](https://ddinter2.scbdd.com/download/)
- [DDInter terms](https://ddinter2.scbdd.com/terms/)
- [PubChem PUG REST documentation](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
- [OpenFDA API documentation](https://open.fda.gov/apis/)
- [ChEMBL web services](https://www.ebi.ac.uk/chembl/api/data/docs)
- [PharmGKB](https://www.pharmgkb.org/)
- [FAERS public dashboard/data](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers)
- [BindingDB](https://www.bindingdb.org/)
- [UniProt REST API](https://www.uniprot.org/help/api)
- [RCSB PDB API](https://www.rcsb.org/docs/programmatic-access/web-apis-overview)
- [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/info/geo_paccess.html)
