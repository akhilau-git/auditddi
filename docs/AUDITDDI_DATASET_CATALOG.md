# AuditDDI Dataset Catalog

## 1. Final dataset decision

AuditDDI uses exactly seven finalized dataset sources:

1. TWOSIDES
2. PubChem
3. ChEMBL
4. openFDA
5. FAERS
6. DDInter
7. PharmGKB

The project does **not** use UniProt, BindingDB, PDB, GEO, DrugCentral, RxNorm, KEGG, DailyMed, DrugBank, or Medi-Span as finalized datasets for the current project.

Some source-code modules still contain optional or historical support for those sources. A module existing in the repository does not mean that its dataset was used in the final training run.

## 2. Current storage layout

The current Colab data root is:

```text
/content/drive/MyDrive/Projects/auditddi/
```

The expected source folders are:

```text
/content/drive/MyDrive/Projects/auditddi/
├── twosides/
├── pubchem/
├── chembl/
├── openfda/
├── faers/
├── ddinter/
└── pharmgkb/
```

The code repository is separate:

```text
/content/drive/MyDrive/Projects/AuditDDI/
```

The old path `/content/drive/MyDrive/auditddi-data` is a legacy location and is not the current project data root.

## 3. Dataset summary table

The counts below describe the project’s recorded source snapshot and intended coverage. Exact counts can change when a different release or filtered export is used. Every training run records input paths, hashes, and row counts in its run artifacts; those run-specific values are authoritative for a final result table.

| Dataset | Main data type | Main role | Known project coverage / size | Used as primary DDI labels? |
|---|---|---|---|---|
| TWOSIDES | Drug-pair adverse-event associations | Primary supervised signal | About 4,649,441 pair-event rows, 645 drugs, 63,473 unordered positive pairs, and 1,317 event types in the recorded project snapshot | Yes |
| PubChem | Compound identity and molecular structure records | Name/identifier-to-SMILES bridge | Used to resolve the project drug catalog; the recorded catalog covers about 645 TWOSIDES drugs | No; structures are model inputs, not DDI labels |
| ChEMBL | Chemical and drug-target annotations | Optional biological feature/evidence source | Recorded target mapping covers roughly 565 project drugs in the current feature artifacts | No |
| openFDA | FDA label and pharmacokinetic/regulatory evidence | Evidence and optional feature source | Recorded CYP-related feature coverage is about 406 drugs in the current feature artifacts | No |
| FAERS | Pharmacovigilance adverse-event reports | Toxicity/evidence ablation source | Release-dependent report tables; row counts must be taken from the FAERS manifest for the exact downloaded snapshot | No |
| DDInter | Curated clinical/literature drug-pair interactions | External validation/agreement source | The recorded source snapshot contains roughly 382,618 interaction records across its downloaded categories | No; kept outside primary training |
| PharmGKB | Drug-gene and pharmacogenomic relationships | Optional biological feature/evidence source | Recorded gene-profile artifacts cover about 241 drugs and 1,233 genes | No |

The words `about`, `roughly`, and `recorded` are intentional: source releases, filters, and mappings can change counts. Do not claim a precise count for a new run unless it is read from that run’s manifest.

## 4. TWOSIDES

### 4.1 What it contains

TWOSIDES is a pharmacovigilance dataset derived from drug-event reporting. Each record describes:

- drug 1 identifier and concept name;
- drug 2 identifier and concept name;
- adverse-event condition identifier and concept name;
- reporting-count fields such as `A`, `B`, `C`, and `D`;
- disproportionality statistics such as `PRR`, `PRR_error`, and `mean_reporting_frequency`.

The raw project file is:

```text
twosides/TWOSIDES.csv.gz
```

The project also stores:

```text
twosides/OFFSIDES.csv.gz
```

OFFSIDES is kept as a separate source/evidence file. It is not silently merged into the primary TWOSIDES training labels.

### 4.2 Raw input format

The raw TWOSIDES file used in the current preparation workflow contains columns such as:

```text
drug_1_rxnorm_id
drug_1_concept_name
drug_2_rxnorm_id
drug_2_concept_name
condition_meddra_id
condition_concept_name
A
B
C
D
PRR
PRR_error
mean_reporting_frequency
```

### 4.3 Prepared training format

The graph training pipeline does not consume the raw RxNorm-name table directly. It consumes a prepared file:

```text
twosides/drug_drug_edges.csv
```

with columns:

```text
source,target,interaction_type
```

Here:

- `source` is the canonical SMILES for drug 1;
- `target` is the canonical SMILES for drug 2;
- `interaction_type` is the normalized adverse-event/condition name.

### 4.4 How TWOSIDES becomes labels

A reported pair-event association becomes positive evidence:

```text
label = 1
label_evidence = reported_twosides
```

The training pipeline canonicalizes pair order and removes duplicate unordered observations. A pair A-B is the same pair as B-A.

TWOSIDES absence is not a negative clinical finding. For binary training, the pipeline samples unreported drug pairs and records them explicitly as sampled unreported examples:

```text
label = 0
label_evidence = unreported_twosides_split_aware_sampled
```

Therefore, a TWOSIDES negative is an unknown/unreported pair, not a confirmed safe pair.

### 4.5 Reproducible name-to-structure preparation

The prepared edge file is generated with
[`scripts/prepare_twosides_edges.py`](../scripts/prepare_twosides_edges.py).
The resolver queries PubChem by the TWOSIDES concept name and then falls back
to the source `drug_*_rxnorm_id` through PubChem's RxNorm cross-reference.
Lookups use bounded retries and a persistent JSON cache. Unresolved names are
written to an audit report and their rows are excluded; no structure is guessed
for products, mixtures, biological materials, or non-chemical concepts that
PubChem cannot resolve. The run summary must record at least the unique-name,
unresolved-name, written-row, and skipped-row counts.

## 5. PubChem

### 5.1 What it contains

PubChem contains chemical compound identity and structure information, including names, compound identifiers, and molecular representations such as SMILES.

### 5.2 How it is used

PubChem is used to map raw TWOSIDES drug concept names to molecular structures:

```text
TWOSIDES drug name -> PubChem lookup -> SMILES -> RDKit canonical SMILES
```

The canonical SMILES is then used to build the molecular graph. PubChem does not supply the binary DDI label in this project.

### 5.3 Input type

The relevant input is a compound identity/name or identifier. The relevant model-preparation output is a valid canonical SMILES string, for example:

```text
CCO
```

Invalid or unresolved structures are audited and excluded rather than replaced with fake structures.

## 6. ChEMBL

### 6.1 What it contains

ChEMBL is a chemical and bioactivity resource. In this project it is relevant to chemical/target annotation and optional biological feature preparation.

### 6.2 How it is used

ChEMBL can produce drug-target mappings and related feature tables. These features are optional architecture inputs and are not part of the default `edge_aware_gat_v2` molecular-only candidate unless explicitly enabled in the run configuration.

### 6.3 Label semantics

A ChEMBL target or bioactivity record is not a DDI label. It describes chemical/biological evidence at the compound-target level.

## 7. openFDA

### 7.1 What it contains

openFDA provides FDA regulatory and Structured Product Label evidence. The project uses it for label/evidence parsing and optional pharmacokinetic or CYP-related feature blocks.

### 7.2 How it is used

openFDA information can be transformed into per-drug evidence or feature vectors. It is not treated as a pairwise positive or negative DDI table for the primary classifier.

### 7.3 Label semantics

An openFDA label mention is regulatory evidence. It is not automatically a confirmed interaction label and is not automatically a safe/non-safe decision.

## 8. FAERS

### 8.1 What it contains

FAERS contains spontaneous adverse-event reports. It can include report-level or aggregated pharmacovigilance information about drugs and adverse events.

### 8.2 How it is used

FAERS is retained as an evidence and auxiliary-toxicity source. It can be used in a separately identified ablation or toxicity analysis. It is not merged into the primary TWOSIDES DDI labels.

### 8.3 Important limitation

Spontaneous reports are subject to reporting bias, confounding, missingness, and lack of causal proof. A FAERS report is not by itself proof that a drug caused an interaction or adverse event.

## 9. DDInter

### 9.1 What it contains

DDInter is a curated interaction resource based on literature and clinical evidence. Records can include drug pairs, interaction categories, severity, and supporting evidence fields.

### 9.2 How it is used

DDInter is kept outside the primary supervised training labels and used for external agreement or transfer-style evaluation. This separation matters because using DDInter as hidden training supervision would make external evaluation invalid.

### 9.3 Label semantics

A DDInter record is curated interaction evidence. Lack of a DDInter record is unknown, not proof of no interaction.

## 10. PharmGKB

### 10.1 What it contains

PharmGKB contains pharmacogenomic relationships involving drugs, genes, enzymes, and related evidence.

### 10.2 How it is used

PharmGKB can produce per-drug gene/enzyme profiles. In optional multimodal or ablation architectures, these profiles can be encoded and fused with molecular graph representations.

In the default edge-aware molecular run, PharmGKB is not automatically a label source and does not change the TWOSIDES label definition.

## 11. What is not part of the final dataset set

The following are excluded from the current finalized dataset contract:

| Excluded source | Reason |
|---|---|
| UniProt | Not a finalized standalone dataset for this project snapshot; any sequence artifacts must be treated as derived/optional artifacts |
| BindingDB | Deferred/optional affinity evidence, not required by the current final run |
| PDB | Not used as a finalized primary source because coverage is sparse |
| GEO | Not used because disease transcriptomics is not the same as drug perturbation data |
| DrugCentral | Not in the finalized seven-source list |
| RxNorm | Raw TWOSIDES identifiers may be present, but RxNorm is not a separate finalized dataset source for this run |
| KEGG | Not in the finalized source list |
| DailyMed | Not in the finalized source list; regulatory evidence is represented through openFDA |
| DrugBank and Medi-Span | Excluded because of access, licensing, and redistribution constraints |

## 12. Source data versus derived artifacts

These are derived artifacts, not additional datasets:

- `drug_drug_edges.csv`;
- canonical SMILES tables;
- molecular graph objects;
- Morgan/ECFP fingerprints;
- split CSV files;
- feature stores;
- model-selection validation tables;
- calibration tables;
- conformal prediction artifacts;
- checkpoints and prediction files;
- run manifests and audit reports.

A derived artifact should always record its source path and, when possible, a SHA-256 hash.

## 13. Dataset-to-model flow

```text
TWOSIDES raw drug names and events
              |
              +-- PubChem identity/structure bridge
              |
              v
canonical drug SMILES and event edge table
              |
              v
RDKit molecular graphs
              |
              v
shared edge-aware GATv2 encoder
              |
              v
symmetric drug-pair representation
              |
              v
binary DDI-risk logit
              |
              v
transductive/S1/S2 metrics and local API checkpoint
```

The other finalized sources remain typed evidence or optional feature sources:

```text
ChEMBL     -> optional target/biological features
PharmGKB   -> optional gene/enzyme profiles
openFDA    -> regulatory/CYP evidence
FAERS      -> pharmacovigilance and toxicity ablation evidence
DDInter    -> external agreement/validation evidence
```

## 14. What to report in the thesis

For every experiment, report:

1. exact dataset release or file snapshot;
2. raw file names;
3. number of raw rows;
4. number of unique drugs;
5. number of unique unordered reported pairs;
6. number of event types;
7. number of rows removed during identity and graph validation;
8. number of positives and sampled unreported negatives;
9. negative-sampling protocol and ratio;
10. model architecture and all enabled feature blocks;
11. model and split seeds;
12. train/validation/test and S1/S2 counts;
13. metrics with confidence intervals;
14. checkpoint hash and run-manifest path.

Do not report a source as "used for training" merely because its folder exists. The run manifest and feature flags must show that the source entered the active computation graph.
