# AuditDDI Dataset and Training Verification

**Verification date:** 2026-10-08  
**Execution environment:** Google Colab with Google Drive mounted  
**Purpose:** Research-only dataset and training-provenance verification

## Scope and storage model

AuditDDI source code is maintained locally and in GitHub. Raw and processed
datasets, Colab notebooks, model artifacts, and experiment outputs are stored
in Google Drive and training is performed in Google Colab. The verified
dataset root was:

```text
/content/drive/MyDrive/auditddi-data
```

The Drive repository checkout was separately present at:

```text
/content/drive/MyDrive/AuditDDI
```

The repository checkout is code and documentation, not an additional dataset
to count with the files under `auditddi-data`.

This report records what was observed in the Drive snapshot. It does not
claim that every historical artifact is part of the selected model run.

## Executive conclusion

The verified Drive snapshot contains a TWOSIDES-derived interaction dataset
with:

| Quantity | Verified value |
|---|---:|
| Drugs in the project catalog | 645 |
| Unique unordered reported drug pairs | 63,473 |
| Pair-event interaction records | 4,649,441 |
| Event types | 1,317 |

The selected seed-11 cross-attention run used the verified TWOSIDES edge file
as its DDI supervision source. Its binary labels were:

- Positive: `reported_twosides`
- Negative: `unreported_twosides_split_aware_sampled`

The negative class is a sampled unreported pair. It is not a confirmed
non-interaction, safe pair, or harmless pair.

The selected run also enabled the project toxicity bridge as an auxiliary
input/task. Other collected sources were not demonstrated as DDI labels in
this run.

## Drive inventory

The non-destructive Drive inventory reported:

```text
Top-level entries under auditddi-data: 80
Files inventoried:                       7,150
```

The inventory included raw/source folders, processed tables, split files,
checkpoints, predictions, notebooks, and repeated experiment-result folders.
Consequently, the sum of all folder row counts is not a dataset sample count:
it would double-count source data and derived artifacts.

Examples of observed folders and physical row/record totals:

| Folder | Files | Physical rows/records | Interpretation |
|---|---:|---:|---|
| `twosides` | 4 | 4,652,720 | TWOSIDES edges, event profile, event names, and drug catalog |
| `unified_graph` | 6 | 4,652,281 | Derived graph tables and bridges |
| `ddinter` | 10 | 382,618 | DDInter source files; separate audit required |
| `chembl` | 2 | 2,915,076 | ChEMBL-derived records |
| `pharmgkb` | 110 | 162,985 | PharmGKB-derived records |
| `BindingDB` | 3 | 14,572 | BindingDB drug, target, and edge tables |
| `PubChem` | 4 | 6,965,049 | PubChem identity/biologics/name tables |
| `event_table_v1` | 5 | 65,435 | Derived sparse/event-table artifacts |
| `p1_targets` | 3 | 1,822 | ChEMBL target preparation |
| `uniprot` | 2,339 | 1,849 parsed records | Protein FASTA and metadata artifacts |
| `faers` | 18 | Not tabularly counted by the first census | FAERS source files |

The inventory also found many result and experiment folders, including
`artifacts`, `auditddi-results*`, `experiments`, `splits`, `checkpoints`, and
historical study folders. These are evidence/artifacts, not independent
datasets.

## TWOSIDES verification

The verified source files were:

```text
twosides/drug_drug_edges.csv
twosides/event_profile.csv
twosides/side_effects.csv
twosides/twosides_drugs.csv
```

### Edge file

`twosides/drug_drug_edges.csv` has:

| Property | Value |
|---|---|
| Rows | 4,649,441 |
| Columns | `source`, `target`, `interaction_type` |
| Unique drugs | 645 |
| Unique unordered pairs | 63,473 |
| SHA-256 | `b05f1c17a8c26966f468ae96a775c8621d6824613aba811f669dc62c8a3bba69` |

The 4,649,441 rows are pair-event associations. They are not 4,649,441
independent binary DDI samples. A single drug pair can occur with many event
types. The binary positive-pair count is 63,473.

For context, 645 drugs permit at most:

```text
645 * 644 / 2 = 207,690
```

unordered pairs. The observed 63,473 positive pairs cover approximately
30.56% of all possible unordered pairs among the project drug catalog.

### Event profile consistency

`twosides/event_profile.csv` contains 1,317 event types. Its `n_pairs` values
sum to 4,649,441.

The event profile was compared with counts computed directly from
`drug_drug_edges.csv`. Raw string comparison initially showed 299 labels on
each side as missing. After the same normalization used by the repository
(Unicode normalization, trimming, lower-casing, and whitespace collapsing),
the comparison passed:

| Check | Result |
|---|---:|
| Raw edge event labels | 1,317 |
| Raw profile event labels | 1,317 |
| Normalized edge event labels | 1,317 |
| Normalized profile event labels | 1,317 |
| Normalized labels missing from profile | 0 |
| Normalized labels missing from edge file | 0 |
| Profile total | 4,649,441 |
| Edge total | 4,649,441 |
| Total difference | 0 |
| Maximum event-count difference | 0 |

Therefore, the event profile is an exact aggregate of the edge table after
documented event-name normalization.

### Derived unified graph

`unified_graph/master_ddi_edges.csv` has:

| Property | Value |
|---|---|
| Rows | 4,649,441 |
| Unique unordered pairs | 63,473 |
| Unique drugs | 645 |
| SHA-256 | `58878e3376ca8db2009bca24f6f3d58670f3d69e903969decbc718dd58a1fbf7` |

The row count and pair count match the TWOSIDES edge table. The hash differs
because the unified graph adds derived fields:

```text
interaction_source
evidence_count
confidence_score
split_group
```

This is a processed representation of the TWOSIDES edge table, not a second
independent set of DDI samples.

The node tables contain 645 rows:

```text
unified_graph/master_drug_nodes.csv
unified_graph/master_drug_nodes_verified_targets.csv
```

The two verified node-table hashes were identical:

```text
85e61ef26c82577c7c74cb29cb3e5161f4eff5f8516c1b869e132458684dc608
```

## Selected training run

The selected run was the seed-11 cross-attention screening run:

| Property | Value |
|---|---|
| Run ID | `20260922T181606Z` |
| Model | `cross_attention_edge_aware_gat_v1` |
| Model seed | 11 |
| Split seed | 42 |
| Repository commit | `ef2861ae756d5c8f098da797e00b45353086495a` |
| Device | Tesla T4 |
| Python | 3.13.15 |
| Evaluation protocol | `standard` |
| Negative protocol | `split_aware_standard_v1` |
| Configured data cap | 200,000 rows |
| Negative label meaning | `unreported_twosides_split_aware_sampled` |
| Conformal alpha | 0.1 |
| Toxicity pair features | Enabled |
| Fingerprint features | Disabled |
| Motif features | Disabled |
| ChemBERTa | Disabled |

The run manifest recorded this TWOSIDES hash:

```text
b05f1c17a8c26966f468ae96a775c8621d6824613aba811f669dc62c8a3bba69
```

It exactly matches the verified Drive edge file.

The manifest also recorded:

```text
toxicity_bridge:
99c2fd75a06f03615874c9b3f43a0bd5277c68abde6fd963d163b037065f4732

training_source:
244742bade5c75f14d48e95fd073605c0831b4cdf6b1695894b2575b68aa570a
```

The `training_source` hash was located and matches:

```text
auditddi/src/training/train_full_pipeline_v2.py
```

It is a source-code hash, not a second dataset hash.

## Selected run split counts

All listed split files were balanced between positive and sampled-unreported
negative rows.

### Transductive view

| Split | Rows | Negative | Positive |
|---|---:|---:|---:|
| Train | 65,232 | 32,616 | 32,616 |
| Test | 18,120 | 9,060 | 9,060 |
| **Train + test** | **83,352** | **41,676** | **41,676** |

The actual transductive binary dataset for this run was therefore 83,352
rows. The configured 200,000 value was a maximum data cap, not the observed
row count.

### S1 both-drugs-unseen view

| Split | Rows | Negative | Positive |
|---|---:|---:|---:|
| Development | 624 | 312 | 312 |
| Test | 620 | 310 | 310 |

### S2 one-drug-unseen view

| Split | Rows | Negative | Positive |
|---|---:|---:|---:|
| Development | 15,532 | 7,766 | 7,766 |
| Test | 15,086 | 7,543 | 7,543 |

### Validation roles

| Split | Rows | Negative | Positive |
|---|---:|---:|---:|
| Validation | 3,624 | 1,812 | 1,812 |
| Post-hoc validation | 3,624 | 1,812 | 1,812 |

S1, S2, transductive, validation, and post-hoc files are separate protocol
roles. Their row counts must not be added together as one dataset size.

## Which datasets were used for training?

### Selected seed-11 run

For the selected run, the evidence supports this data-use statement:

| Role | Source/status |
|---|---|
| Positive DDI supervision | TWOSIDES reported pair/event records |
| Negative supervision | Split-aware sampled unreported TWOSIDES pairs |
| Molecular inputs | Structures associated with TWOSIDES project drugs |
| Auxiliary toxicity input/task | Project FAERS-derived toxicity bridge enabled |
| DDInter DDI labels | Not demonstrated in this run |
| BindingDB features | Not enabled in this run |
| PharmGKB features | Not enabled in this run |
| GEO features | Not enabled in this run |
| DrugBank | Not used; no verified DrugBank data folder was present |

The most accurate description is:

> The selected model was trained for binary DDI prediction using TWOSIDES-
> derived reported pairs and split-aware sampled unreported pairs, with
> molecular graph inputs and the project toxicity bridge as an auxiliary
> input/task. Other collected sources were not demonstrated as DDI labels or
> inputs to this selected run.

### Other project pipelines

The repository contains separate preparation and experimental paths for
additional sources:

| Source | Project role | Proven as an input to selected run? |
|---|---|---|
| TWOSIDES | Main DDI labels and split construction | Yes |
| DDInter | Curated complementary DDI evidence/importer | No |
| DrugBank | Licensed reference source, legally gated | No |
| PubChem | Names, identifiers, and structure bridge | Not as DDI labels |
| FAERS | Toxicity/adverse-event bridge | Yes, auxiliary bridge |
| BindingDB | Drug-target/binding features | No |
| PharmGKB | Gene/enzyme profiles | No |
| ChEMBL | Target/mechanism and pretraining paths | No |
| UniProt | Protein sequence/target metadata | No |
| PDB | Structural target features | No |
| GEO | Transcriptomic signatures | No |

The existence of an importer, folder, notebook, or processed artifact does
not prove that the source was consumed by a particular training run.

## Source inventory facts outside TWOSIDES

These counts were observed during the Drive audit but are not DDI sample
counts for the selected run:

| Source/file | Observed count | Meaning |
|---|---:|---|
| `BindingDB/drug_target_edges.csv` | 8,551 rows | Drug-target records |
| `BindingDB/bindingdb_drugs.csv` | 5,915 rows | BindingDB drug records |
| `BindingDB/bindingdb_targets.csv` | 106 rows | BindingDB target records |
| `PubChem/CID-Biologics.tsv.gz` | 6,869,436 rows | PubChem biologics table |
| `PubChem/Drug-Names.tsv.gz` | 92,519 rows | PubChem names table |
| `uniprot/master_nodes_with_uniprot.csv` | 645 rows | Derived project drug nodes |
| `uniprot/uniprot_targets_metadata.csv` | 1,204 rows | Protein-target metadata |
| `unified_graph/faers_bridge.csv` | 1,500 rows | Derived FAERS toxicity bridge |

These source-specific counts must not be summed with the TWOSIDES counts.

## Provenance and completeness status

### Verified

- Drive is mounted and both Drive paths exist.
- TWOSIDES edge file exists and has a fixed SHA-256.
- TWOSIDES has 645 drugs, 63,473 unique pairs, 4,649,441 pair-event rows,
  and 1,317 event types.
- Event-profile totals agree exactly after normalization.
- Selected run's TWOSIDES hash matches the verified edge file.
- Selected run's repository commit, model seed, split seed, and runtime are
  recorded.
- Split row counts, class counts, and split-file hashes are recorded.
- The training-source hash resolves to `train_full_pipeline_v2.py`.

### Limitations

- The selected copied result directory contains
  `run_manifest_initial.json`, but the final `run_manifest.json` was not
  present at that path.
- The exact official/original TWOSIDES release size was not independently
  verified. The verified 4,649,441-row file may already be filtered or
  transformed.
- DDInter, ChEMBL, PharmGKB, FAERS, BindingDB, PubChem, UniProt, PDB, and GEO
  require separate source-specific audits if they are to be claimed as
  training inputs or independent evaluation datasets.
- A Drive inventory is not evidence of legal permission to use DrugBank.
  DrugBank must remain excluded until written permission covers the intended
  use.
- The selected run's result copy does not by itself establish repeated-seed
  performance or successful S1 generalization.

## Final verified statement

> AuditDDI's verified Drive snapshot contains a TWOSIDES-derived dataset with
> 645 drugs, 63,473 unique unordered reported drug pairs, 4,649,441
> pair-event associations, and 1,317 event types. The selected seed-11
> cross-attention run used the verified TWOSIDES edge file and the
> split-aware sampled-unreported protocol. It produced 65,232 transductive
> training rows and 18,120 transductive test rows, with S2 and S1 test sets
> of 15,086 and 620 rows respectively. Its DDI supervision was TWOSIDES
> based; a project FAERS-derived toxicity bridge was enabled as an auxiliary
> input/task. Other collected sources were not demonstrated as inputs to this
> selected run. Sampled unreported pairs are unknown labels, not confirmed
> safe or non-interacting pairs.

## Related implementation

- [Project README](../README.md)
- [End-to-end project workflow](AUDITDDI_END_TO_END_PROJECT_WORKFLOW.md)
- [Training pipeline](../src/training/train_full_pipeline_v2.py)
- [Pair split construction](../src/data_prep/splits.py)
- [Event-table construction](../src/data_prep/build_event_table.py)
- [Unified graph construction](../src/data_prep/build_unified_graph.py)
