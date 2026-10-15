# AuditDDI

AuditDDI is a research project for predicting **reported drug–drug interaction (DDI) associations** from molecular structure and linked drug evidence. Training and evaluation run in Google Colab; a reviewed model checkpoint can then be served locally through a FastAPI backend and web frontend.

> **Research prototype:** a model score is not medical advice or a clinical probability. The training labels distinguish reported TWOSIDES pairs from sampled pairs that were not reported in the supplied data. An unreported pair is unknown; it is not known to be safe.

## Project at a glance

```text
Seven source folders in Google Drive
        ↓
Drug identity resolution and source audits
        ↓
Molecular and drug-profile feature preparation
        ↓
ChEMBL encoder pretraining → TWOSIDES-supervised DDI training
        ↓
Transductive, S1, and S2 evaluation + run artifacts
        ↓
Review a checkpoint → local FastAPI service + web app
```

The repository contains source code, preparation scripts, the model, evaluation tools, a local application, and tests. Raw datasets, generated artifacts, and model checkpoints are kept outside Git—normally in Google Drive or local deployment storage.

## The seven datasets and their roles

The project uses these seven sources. They have different jobs; their records are **not concatenated into one training-label table**.

| Source | Role in AuditDDI |
|---|---|
| **TWOSIDES** | Main supervised pair labels: reported drug pairs are positive examples. |
| **PubChem** | Drug identity and chemical structure resolution; canonical SMILES are used to build molecular inputs. |
| **ChEMBL** | Chemical structures for leakage-audited, self-supervised molecular encoder pretraining. |
| **DDInter** | Curated interaction evidence; known pairs are excluded from sampled negatives and retained for audit. Unlisted pairs remain unknown. |
| **PharmGKB** | Matched drug–gene evidence used in the drug-profile feature bundle. |
| **openFDA** | Matched label-derived CYP profile features in the source bundle. |
| **FAERS** | Pharmacovigilance coverage and reporting-bias audit. FAERS report counts are not model input features in the current workflow. |

The current multimodal candidate uses a shared edge-aware GATv2 molecular encoder, Morgan fingerprints, and a 51-dimensional audited drug profile. The ChEMBL-pretrained encoder initializes the molecular encoder. TWOSIDES remains the source of the model's supervised DDI labels; the other sources contribute structures, features, pretraining, or separate evidence/audits according to the table above.

## Repository layout

```text
backend/                 FastAPI service and local authentication
frontend/                Browser application served by Nginx
scripts/                 Dataset checks and preparation commands
src/data_prep/           Drug resolution, graph preparation, source linking
src/models/              Molecular encoders, pair model, calibration and audits
src/training/            ChEMBL pretraining, DDI training, evaluation and studies
tests/                   Automated project tests
docs/                    Dataset, methodology, and workflow notes
requirements.txt         Local/API Python dependencies
requirements_colab.txt   Colab dependencies; leaves Colab's PyTorch build in place
docker-compose.yml       Local two-service deployment
```

Get the source code from the project repository:

```bash
git clone https://github.com/akhilau-git/auditddi.git
cd auditddi
```

## Data and artifacts

For the current Colab workflow, set the Drive project root to:

```text
/content/drive/MyDrive/Projects/auditddi
```

Expected source folders are `twosides/`, `pubchem/`, `chembl/`, `pharmgkb/`, `openfda/`, `faers/`, and `ddinter/`. Prepared inputs and generated artifacts live beside them:

```text
auditddi/
├── twosides/ pubchem/ chembl/ pharmgkb/ openfda/ faers/ ddinter/
├── processed/
│   ├── twosides_v1/
│   ├── auditddi_v1/
│   └── auditddi_v2/
├── checkpoints/
└── results/
```

The code checkout may be stored at `auditddi/code/` in Drive or elsewhere. Keep the code root and data root distinct in configuration: `AUDITDDI_DATA_BASE` points to the Drive project folder; commands are run from the repository root.

## Colab workflow

Use a GPU runtime when available. The trainer selects CUDA automatically and falls back to CPU when CUDA is unavailable; CPU runs can take substantially longer. Mount Drive before running these steps. Run commands from the repository root, such as `/content/auditddi` after cloning, while keeping the datasets under the Drive data root above.

### 1. Install Colab dependencies

From the repository root:

```python
%pip install -r requirements_colab.txt
```

PyTorch is intentionally not installed by this file, preserving Colab's existing PyTorch/CUDA build. If a dependency install requires a runtime restart, restart first, remount Drive, and reapply environment variables.

### 2. Verify the prepared data

The current workflow expects the audited TWOSIDES preparation and identity audits under `processed/`. Verify an existing Drive snapshot without downloading or modifying it:

```bash
python scripts/verify_existing_datasets.py \
  --data-root /content/drive/MyDrive/Projects/auditddi
```

The finalized TWOSIDES pair table is `processed/twosides_v1/twosides_clean.parquet`, with its drug catalogue in `processed/twosides_v1/drugs_clean.csv`.

### 3. Build the seven-source feature bundle

```bash
python scripts/prepare_auditddi_source_bundle.py \
  --data-root /content/drive/MyDrive/Projects/auditddi \
  --parse-openfda
```

This produces `processed/auditddi_v2/`, including the source-feature matrix and bundle manifest. The multimodal trainer checks that required prepared files and source evidence are present; inspect `source_bundle_manifest.json` before training.

### 4. Pretrain the molecular encoder on ChEMBL

Set the paths so the encoder checkpoint and run artifacts persist in Drive:

```python
import os
from pathlib import Path

PROJECT = Path("/content/drive/MyDrive/Projects/auditddi")
os.environ["AUDITDDI_DATA_BASE"] = str(PROJECT)
os.environ["AUDITDDI_RESULTS_BASE"] = str(PROJECT / "results")
os.environ["AUDITDDI_PRETRAIN_CHECKPOINT_PATH"] = str(
    PROJECT / "checkpoints/chembl_pretrained_encoder.pt"
)
os.environ["AUDITDDI_PRETRAINED_ENCODER_PATH"] = str(
    PROJECT / "checkpoints/chembl_pretrained_encoder.pt"
)
os.environ["AUDITDDI_MODEL_ARCHITECTURE"] = "auditddi_multimodal_v1"
```

Then run from the repository root:

```bash
python -u src/training/pretrain_chembl_encoder.py
```

This is a separate self-supervised stage. It writes an encoder checkpoint and a pretraining manifest/history; it does **not** train the DDI classifier. If the audited ChEMBL checkpoint already exists and matches the current inputs, do not repeat pretraining unnecessarily.

### 5. Train and evaluate the multimodal DDI model

Configure the run and keep the outputs in Drive:

```python
import os
from pathlib import Path

PROJECT = Path("/content/drive/MyDrive/Projects/auditddi")
RESULTS = PROJECT / "results/seed_201"
RESULTS.mkdir(parents=True, exist_ok=True)

os.environ["AUDITDDI_DATA_BASE"] = str(PROJECT)
os.environ["AUDITDDI_RESULTS_BASE"] = str(RESULTS)
os.environ["AUDITDDI_MODEL_ARCHITECTURE"] = "auditddi_multimodal_v1"
os.environ["AUDITDDI_MODEL_SEED"] = "201"
os.environ["AUDITDDI_SPLIT_SEED"] = "42"
os.environ["AUDITDDI_EPOCHS"] = "200"
os.environ["AUDITDDI_PRETRAINED_ENCODER_PATH"] = str(
    PROJECT / "checkpoints/chembl_pretrained_encoder.pt"
)
os.environ["AUDITDDI_PUBLISH_LATEST_RESULTS"] = "false"
```

Run:

```bash
python -u src/training/train_full_pipeline_v2.py
```

Training uses TWOSIDES pair labels, split-aware sampled unreported pairs, and a binary cross-entropy-with-logits objective. The trainer selects the best checkpoint on its model-selection validation split and applies early stopping. Each timestamped run writes manifests, split/input audits, training history, metrics, predictions, plots, and checkpoint metadata under the configured results location. Preserve that run folder with its checkpoint when reporting results.

### 6. Understand and report the evaluation

The pipeline reports three evaluation regimes:

- **Transductive:** both drugs are represented in training.
- **S1:** both drugs are unseen during training.
- **S2:** one drug is unseen during training.

Metrics include AUROC, PR-AUC/average precision, F1, MCC, balanced accuracy, and calibration-related measures. Use the run's final manifest and evaluation artifacts for reported numbers; a checkpoint's stored validation AUROC is a model-selection value, not a test result. A sampled unreported pair is not a confirmed-safe negative, and internal metrics are not clinical validation.

For more detail on split construction, leakage controls, and artifacts, see [`docs/AUDITDDI_METHODOLOGY_AND_WORKFLOW.md`](docs/AUDITDDI_METHODOLOGY_AND_WORKFLOW.md).

### Interrupted runs and warm starts

Training checkpoints normally preserve the best model weights; they do not preserve the complete optimizer state needed for exact epoch-by-epoch continuation. The optional warm-start settings in `train_full_pipeline_v2.py` load a provenance-matched checkpoint and continue with newly initialized optimizer state:

```text
AUDITDDI_RESUME_CHECKPOINT_PATH=/path/to/saved_model.pt
AUDITDDI_RESUME_AFTER_EPOCH=<last completed epoch shown in the log>
AUDITDDI_CHECKPOINT_PATH=/path/to/a/new/output_checkpoint.pt
```

This is a warm start from saved weights, not an exact resume. Keep the input checkpoint and output checkpoint paths different, and use the same data, model configuration, and split seed. The trainer validates the saved input/split provenance before continuing.

## Local deployment

The local application loads a reviewed checkpoint; it does not train the model. Copy the selected checkpoint into `backend/checkpoints/` and name it `pxddi_model.pt`, or select another filename below that directory with `PXDDI_CHECKPOINT_PATH`. Keep the corresponding run manifest and evaluation artifacts for provenance. Do not deploy a checkpoint solely because its validation score is high.

### Run with Docker Compose

1. Place the reviewed checkpoint in `backend/checkpoints/pxddi_model.pt`.
2. Create a local `.env` file with a private bootstrap token for the first administrator account:

   ```dotenv
   AUDITDDI_AUTH_BOOTSTRAP_TOKEN=replace-with-a-long-random-one-time-token
   ```

3. From the repository root, build and start the application:

   ```bash
   docker compose up --build
   ```

4. Open `http://localhost:3000/login.html`, create the initial administrator using the bootstrap token, and then use the admin page to create accounts and assign roles.

Compose serves the frontend at `http://localhost:3000` and binds the backend API to `127.0.0.1:8000`. The frontend proxies its API requests to the backend. Check `http://localhost:8000/health` and `http://localhost:8000/ready` to inspect service status.

Useful API routes include `POST /predict`, `POST /explain` when supported by the selected model, and `POST /audit/dossier`. Predictions are research estimates and explicitly retain the TWOSIDES label caveat.

For local development without Docker, install `requirements.txt`, put the checkpoint in `backend/checkpoints/`, then start the API from the repository root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

The backend expects its checkpoint inside `backend/checkpoints/`. See `.env.production.example` for the configuration requirements if adapting the service to a hosted environment. A public deployment also needs infrastructure-managed HTTPS, authentication, shared rate limiting, monitoring, and durable audit logging; the local Compose setup alone does not provide those services.

## Development

Install the development dependencies and run the test suite from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Do not commit raw datasets, credentials, local `.env` files, or trained checkpoints. Keep generated run outputs and model artifacts in the designated Drive/results or local checkpoint folders.

## Limitations

- TWOSIDES is observational pharmacovigilance evidence and may contain reporting bias.
- An unreported pair is unknown, not proven safe.
- S1/S2 cold-start performance can differ substantially from transductive performance.
- Calibration, conformal sets, and structural-domain flags are research diagnostics; they do not guarantee reliability under distribution shift.
- The project does not use patient-linked outcome data and is not a patient-specific clinical decision system.
