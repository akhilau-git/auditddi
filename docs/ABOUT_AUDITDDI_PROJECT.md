# About AuditDDI: project story, current status, and path to completion

**Project name:** AuditDDI — *An Auditable Graph-Fingerprint Framework for Cold-Start Drug-Drug Interaction Prediction*  
**Repository:** [akhilau-git/auditddi](https://github.com/akhilau-git/auditddi)  
**Work split:** model training and datasets in Google Colab/Google Drive; application and final demo run locally.  
**Status date:** 6 October 2026 (repository review; Drive contents and external account permissions have not been independently inspected in this review).

## 1. The idea in simple words

Imagine we have two medicine names. AuditDDI is meant to help a researcher ask: “Has this pair been reported to interact, what evidence do we have, how sure is the model, and have these medicines appeared in the model's training data before?”

The computer cannot safely understand a medicine from its name alone. First it must resolve the name to the exact chemical form and structure. Then the model can compare the two chemical structures and other permitted evidence. Finally, the app should show both the model's estimate and the sources that support or disagree with it.

The special research challenge is **cold start**: one or both drugs may be new to the model. A model that works only when it has already seen both drugs is not enough for the project goal. AuditDDI should report when a drug or pair is outside its training experience and should be able to abstain or show uncertainty.

This is a research tool. It is not a doctor, a prescription checker, or a clinically validated safety system.

## 2. How the project began and how its scope grew

Based on the project history shared by the project owner and the files currently in the repository, the work developed in these stages:

1. **Start with a DDI prediction project.** The initial goal was to use machine learning on drug pairs and build a working local application.
2. **Make the goal about unseen drugs.** The project became focused on cold-start prediction and explaining/auditing why a prediction should be reviewed. This is why the name includes “Auditable” and “Cold-Start.”
3. **Build the first data and model path.** TWOSIDES-derived interaction records became the main current training signal. Structure-based graph and fingerprint models, data splitting, and evaluation code were developed around that signal.
4. **Add other evidence sources.** The project grew to include or plan sources such as DDInter, PubChem, ChEMBL, PharmGKB, FAERS, BindingDB, UniProt, PDB, GEO, OpenFDA, and DrugBank. These sources do not all describe the same thing; some identify chemicals, some describe targets, some report safety evidence, and some provide interaction records.
5. **Build the app around the research code.** A FastAPI backend and web dashboard were added so that a local user can submit drug structures and see a research result.
6. **Audit the difference between “code exists” and “project is complete.”** Project review identified that an importer or API helper is not the same as a dataset being downloaded, mapped to the correct drugs, used in training, validated, and displayed correctly. It also identified that cold-start results need to be honestly reported.
7. **Plan the Colab and Drive workflow.** The owner specified that the datasets and training happen in Colab using the Google Drive `auditddi-data` folder, while the application is run locally. This is the intended operating plan.
8. **Plan DrugBank access.** The owner contacted DrugBank about permitted use. Downloading or integrating DrugBank data is waiting on written permission and usable account access. It is not complete merely because a login, an API example, or a downloader script exists.
9. **Document the end-to-end workflow.** A detailed workflow was written at [AUDITDDI_END_TO_END_PROJECT_WORKFLOW.md](AUDITDDI_END_TO_END_PROJECT_WORKFLOW.md). This file explains the project status and history in one place.

This history is a plain-language summary of the conversation and repository, not a claim that every historical activity or Drive file has been independently verified.

## 3. What is in the project today

### 3.1 Research and model code

The repository contains code for molecular graph/fingerprint preparation, graph neural-network architectures, baselines, multimodal/auxiliary evidence components, uncertainty and calibration, explanation audits, cold-start/scaffold evaluation, experiment artifacts, and training scripts. Examples include:

- `src/models/` — DDI models, encoders, calibration, uncertainty, applicability-domain, neighbor-memory, and explanation components.
- `src/data_prep/` — TWOSIDES preparation, DDInter importing, PubChem bridging, and source-specific PharmGKB, FAERS, ChEMBL, BindingDB, UniProt, PDB, GEO, and graph-preparation code.
- `src/training/` — full-pipeline training, multitask, baseline, ensemble, and evaluation scripts.
- `src/evaluation/` — DDI metrics, clinical-audit formatting, degradation and applicability analyses.

These files show that the project has substantial research/software work. They do **not**, by themselves, prove that every source has been downloaded, that every bridge is populated, or that each source is consumed in the final training run.

### 3.2 Data and labels

The current training entry point `src/training/train_full_pipeline_v2.py` reads TWOSIDES-derived edges and samples unreported pairs under a documented protocol. The label means “reported in TWOSIDES” versus “sampled unreported pair”; it does **not** mean “harmful” versus “safe.” The missing/unreported pair is not proof that no interaction exists.

The repository includes `src/data_prep/ddinter_pipeline.py`. It can download/normalize DDInter category CSV exports and records that the normalized data are source evidence. It explicitly says the result is **not evaluation-ready** until identity mapping, overlap auditing, and a defensible negative-label source exist. Importing DDInter is not yet the same as training on DDInter.

The repository has source-specific preparation code for more datasets, but each still needs an inventory and audit: exact release/files, license, schema, drug identifier coverage, matching rate, duplicate/conflict handling, and whether/how it affects model inputs or labels.

The owner reports that the Google Drive folder `auditddi-data` contains datasets and Colab notebooks/results. The Drive itself was not opened or verified in this repository review. Its precise contents, current model runs, and data licenses must be inventoried in Colab before saying which sources are ready.

### 3.3 Model results and what they mean

The repository README records a committed reference checkpoint with internal validation metadata (validation AUROC 0.8972, selected threshold 0.4404). These are not external validation or clinical results. The README also records one archived GPU screening run: transductive AUROC 0.9161, S2 (one drug unseen) AUROC 0.6698, and S1 (both drugs unseen) AUROC 0.5003. It reports conformal coverage falling under cold-start shift.

Those figures matter because they tell us where the project stands: the system has a working research model path and useful evaluation code, but the hardest both-unseen goal is not yet demonstrated. The S1 value is near chance in the cited single run. We need fresh, reproducible, multi-seed evidence before claiming the model is good at predicting interactions for two unseen drugs.

The README also says the committed reference checkpoint predates the current auxiliary-toxicity loss record. It can be loaded as a DDI research reference, but it must not be used to claim that the newer auxiliary toxicity task was successfully trained. New candidates should use current code and their own complete run manifests.

### 3.4 Backend and frontend

The repository contains a FastAPI backend in `backend/main.py` with, among others, `/predict`, `/explain`, audit dossier routes, `/health`, and `/ready`. The prediction path accepts structures (SMILES), validates them, and runs the loaded model. That is a structure-based prediction interface; it is not yet a complete “type any drug name and resolve every source” workflow.

The frontend contains the research dashboard, and Docker Compose runs the web frontend locally at `http://localhost:3000` and binds the backend privately on local port 8000. The README gives the local development and Colab training instructions. Code and configuration for an app do not prove a successful live end-to-end interaction on the owner's computer; that still needs an actual local run with the selected checkpoint and real request.

### 3.5 DrugBank status

The repository contains a DrugBank API helper and a bulk collection script. The bulk script is not evidence that DrugBank records were collected. It assumes DrugBank IDs can be extracted from the DDInter normalized file and calls a route that must be checked against official documentation and the account's approved product/access. DDInter identifiers and DrugBank identifiers are not interchangeable by assumption.

The owner says an email has been sent to DrugBank. Until the written reply confirms the project's specific uses, the DrugBank training, benchmarking, bulk-download, retention, and display parts are **pending**. Do not put DrugBank account credentials into the repository or a shared notebook. The official release page can also change which files are available. We must inspect actual authorized release metadata after permission/access is confirmed rather than hard-code a guessed URL.

### 3.6 Git and data separation

The intended code repository is GitHub. Google Drive is the storage location for private/large datasets and training outputs. The project `.gitignore` already excludes environment files and many outputs/checkpoints; this review added explicit ignore patterns for the planned Drive data/results folders and DrugBank raw/cache folders. Every future commit still needs a staged-file review. A `.gitignore` rule is not a substitute for checking `git status`.

## 4. How much is complete?

There is no trustworthy single percent for the whole project. “Code is present,” “dataset has been downloaded,” “model trained,” “unseen-drug performance demonstrated,” and “app connected” are different completion checks. Saying “92–95% complete” would hide important unfinished research and deployment work.

Here is the more honest status by work area, based on the repository snapshot reviewed on 6 October 2026 and the owner's statements:

| Work area | Current status | What supports this status | What remains |
|---|---|---|---|
| Project idea and scope | **Defined** | Name, cold-start and auditability goals in README and user instructions | Keep all reports and interface wording consistent with research-only scope |
| Repository and research code | **Substantial code present** | Model, data-prep, training and evaluation modules exist | Review each pipeline, resolve gaps, reproduce outputs from a clean Colab run |
| TWOSIDES training path | **Implemented; archived runs exist** | Current training script reads TWOSIDES; README records a checkpoint and a GPU screening run | Re-run under current code, multiple seeds, and confirm Drive inputs and artifacts |
| DDInter data preparation | **Importer present; training integration pending** | Importer normalizes source files and marks them not evaluation-ready | Download permitted files to Drive, inspect schema, map identities, define source-specific training/evaluation protocol |
| Other source pipelines | **Partly implemented/planned; source-by-source status unverified** | Preparation modules exist for multiple sources | Verify actual data, terms, mappings, coverage, quality, and model consumption for each source |
| DrugBank | **Blocked/pending permission and access** | Helper/collector code exists; owner says email sent | Written use confirmation, available account files/API, correct ID crosswalk, authorized collection and retention, reviewed integration |
| Drug identity and structure resolution | **Partial** | Structure processing and PubChem bridge modules exist; model needs SMILES | Build a verified cross-source identity map and user-facing candidate selection; audit ambiguous mappings |
| Cold-start model performance | **Not yet demonstrated for the hardest target** | Archived one-seed S1 AUROC is about 0.50; S2 is about 0.67 | Improve and independently re-evaluate with leakage-safe repeated cold-drug/scaffold splits; report limits honestly |
| Severity prediction | **Not established** | DDInter importer can retain a severity field; current principal training task is binary TWOSIDES signal | Define and train a separate severity task with supported labels and suitable validation before showing severity scores |
| API/backend | **Research routes implemented** | FastAPI endpoints and readiness/health code are present | Run locally with the chosen checkpoint; wire any name search and permitted live evidence routes intentionally |
| Frontend | **Dashboard code present** | Frontend directory and Compose service exist | Verify search → exact compound → structure → prediction → evidence display end-to-end |
| Local deployment | **Documented/configured; owner run not verified here** | README and Docker Compose specify local operation | Start services on owner's machine and confirm health, checkpoint, UI response, and logs |
| Workflow documentation | **Written** | End-to-end workflow document added | Update as permissions, Drive inventory, code and training status change |
| GitHub safety | **Improved; ongoing check required** | Ignore rules include local env/results/checkpoints and added Drive/DrugBank paths | Review every staged commit; never push licensed datasets, secrets or checkpoints |

## 5. Where we are right now

We are at the **integration and evidence-validation stage**, not at the “everything is done” stage.

In simple words:

- The project has a substantial toolbox and a local app foundation.
- TWOSIDES currently supplies the main model-training labels.
- DDInter can be prepared, but the current training script does not automatically learn from that import.
- Several other source connectors/preparation scripts exist, but each one must be checked with the actual Drive files and their rules.
- The hardest cold-start result, when both medicines are unseen, needs major improvement and a fresh repeatable evaluation.
- DrugBank is waiting for written permission and verified access. A code file is not access, and access is not automatically permission for every proposed use.
- The requested full connected data graph, multicategory/severity task, and verified local UI flow remain work to do.

## 6. What must happen next, in order

### Step 1 — Confirm the project inputs in Drive

In Colab, mount the owner's Drive and list the `auditddi-data` folders, actual filenames, sizes, release/readme/license files, and notebook/run outputs. Do not guess that every planned dataset exists. Save a private inventory/manifest in Drive and share only a non-sensitive summary for review.

**Done when:** we can name the exact source files, versions, formats, and terms that are available, and the training notebook/code revision is known.

### Step 2 — Freeze and label the current baseline

Identify the exact current TWOSIDES source and the checkpoint/run that produced each reported metric. Record code commit, input hashes, seeds, split IDs, package/GPU versions, model architecture, checkpoint checksum, and metrics. Re-run the baseline under current code in Colab.

**Done when:** a clean, reproducible baseline run produces a complete manifest and saved predictions for transductive, S1 and S2 evaluation.

### Step 3 — Finish DDInter preparation and audit

Download the official DDInter files allowed by its terms, keep the raw files in Drive, run the importer, inspect the actual schema and severity labels, and map DDInter identities to the canonical compound registry. Count overlap with TWOSIDES and report unresolved names/IDs. Keep DDInter missing pairs unknown.

**Done when:** the source is reproducibly imported and a matching/overlap report exists. This still does not mean the training integration is complete.

### Step 4 — Resolve DrugBank permission

Wait for the reply to the already-sent DrugBank email. If the reply is unclear, ask what is permitted for academic training, benchmarking, Drive retention, and student-prototype display. If those uses are not permitted, do not use DrugBank for them; proceed with other permitted sources.

**Done when:** written scope is recorded and the account can retrieve an authorized release/API resource. If that never happens, mark DrugBank “not used” and continue without it.

### Step 5 — Build identity mapping before combining sources

Create one internal drug registry and source-ID crosswalk. Resolve exact forms using stable IDs and chemistry identifiers such as InChIKey/CID where available. Review multiple candidate matches by hand. Preserve original identifiers, source, mapping method, and confidence. Do not merge by drug-name spelling alone.

**Done when:** every source has a coverage table for matched, unmatched, ambiguous, invalid and excluded drugs, and the mapping can be regenerated.

### Step 6 — Design the multi-source evidence graph

Store interaction labels separately from structure, drug-target, pharmacogenomic, label-text and adverse-event evidence. Record provenance on each fact. Keep TWOSIDES reported/unreported sampling, DDInter curated records/severity, and any permitted DrugBank facts as different evidence types. Use masked/source-specific learning tasks if meanings differ.

**Done when:** a data dictionary, conflict policy and source-specific label definitions are documented and implemented. Never turn every unlisted pair into a negative.

### Step 7 — Connect only appropriate data to model tasks

Use TWOSIDES as the reproducible baseline. Then implement and compare a DDInter task/ablation after identity mapping and label review. If severity is trained, use a separate ordinal or otherwise justified target and evaluate it separately. Use molecular/target/pharmacogenomic features only where coverage and mapping are good. Keep DrugBank out unless its permission explicitly covers each use.

**Done when:** the training loader reports source counts and the run manifest proves which sources, features, and labels were actually used.

### Step 8 — Train and evaluate in Colab

Use GPU Colab as planned. Fix seeds and splits; prevent the same canonical pair from crossing train/test across sources. Report repeated seeds and transductive, one-unseen, both-unseen, scaffold and (when dated data allow) temporal results. Report AUROC and AUPRC plus calibration/uncertainty and per-source counts. For severity, use appropriate ordinal measures and class-level results.

**Done when:** selected model improves the cold-start goal over documented baselines on independent, leakage-safe evaluations, or the report explicitly states that the goal was not achieved. Do not hide a weak S1 result behind a high transductive result.

### Step 9 — Promote a reviewed model to the local backend

Copy the selected checkpoint and every required inference artifact from Drive to the local ignored checkpoint directory. Record exactly which run it came from. Make backend model metadata match the model architecture and feature schema. Keep DrugBank tokens on the backend only if authorized and configured.

**Done when:** backend readiness identifies the intended model, and a local prediction uses that checkpoint successfully.

### Step 10 — Finish the user flow in the local frontend

Search for a typed drug name through permitted sources, let the user select the exact compound, validate/resolve its SMILES, call the model, and show uncertainty/domain status. Show external source evidence in a separate panel with source and date. Clearly show missing evidence and conflicts. Do not call the model score an official DrugBank result or a clinical severity estimate.

**Done when:** a local person can follow the entire flow and each displayed fact can be traced to the selected compound, checkpoint, or source record.

### Step 11 — Local run and handoff

Run the backend and frontend locally (or use Compose), check health/readiness, send a known example through the UI, inspect the result and logs, and record the app/model version. Keep it local unless security, permission, and validation requirements for a public deployment are met.

**Done when:** the actual local setup works from documented steps on the intended computer.

### Step 12 — Review and push the code

Review every staged path. Push only source code and documentation that are allowed to be shared. Do not push Drive datasets, DrugBank materials, results/checkpoints, passwords, tokens, `.env`, or private user data. Keep the Drive training folder and GitHub code repository separate.

**Done when:** GitHub has the reviewed code revision and the matching run manifest identifies that revision.

## 7. What “fully complete” means for this project

AuditDDI is complete as a **reproducible research prototype** only when all the items below are true or honestly marked “not applicable/not permitted”:

- [ ] The Drive inventory, source versions and terms are recorded.
- [ ] Raw files are preserved; processed files are reproducible from code.
- [ ] Drug identity and structure matching are measured and ambiguous cases reviewed.
- [ ] Each dataset has a declared role; evidence types and label meanings are not mixed up.
- [ ] The model loader actually consumes every source claimed in the paper/demo.
- [ ] The model is evaluated with leakage-safe splits, including both-unseen drug pairs, repeated seeds and baselines.
- [ ] Severity is shown only if a distinct severity model has valid labels and appropriate evaluation.
- [ ] Uncertainty/applicability warnings describe their actual limitations under distribution shift.
- [ ] The approved checkpoint and all required inference metadata load locally.
- [ ] Drug-name search, exact compound selection, structure resolution, model prediction and evidence display work end-to-end.
- [ ] DrugBank is used only in the exact ways confirmed in writing; if not approved, it is not used for those stages.
- [ ] GitHub contains no credentials, licensed raw data, private Drive contents or model binaries.
- [ ] The report describes weak results, source conflicts, missing coverage and limitations plainly.

“Fully complete” here means the research prototype is reproducible and accurately represented. It does not mean clinically validated or safe for treatment decisions.

## 8. The next action

The next practical action is **inventory the actual Drive folder in Colab**, then share the non-sensitive folder/file list and the latest training run summary. That tells us which pipelines can be connected now and which source files, schemas, identifiers, or permissions are still missing. The DrugBank portion waits for its written response; the rest of the allowed-source inventory and baseline review can proceed meanwhile.

## 9. Related documents and official references

- [End-to-end workflow and commands](AUDITDDI_END_TO_END_PROJECT_WORKFLOW.md)
- [Project README and current model/deployment notes](../README.md)
- [DrugBank download help](https://portal.drugbank.com/releases/help)
- [DrugBank latest release page](https://go.drugbank.com/releases/latest)
- [DrugBank Discovery API documentation](https://docs.drugbank.com/discovery/v1/)
- [DDInter downloads](https://ddinter2.scbdd.com/download/)
- [DDInter terms](https://ddinter2.scbdd.com/terms/)
- [PubChem PUG REST documentation](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
- [OpenFDA API documentation](https://open.fda.gov/apis/)
