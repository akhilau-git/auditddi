# Leakage-aware multimodal training in Colab

This protocol uses the collected sources according to what each one can safely
teach the model. A dataset folder existing in Drive does not mean its records
have entered a training run. Every run must preserve the prepared master-node
table, split CSVs, their hashes, modality coverage, the model seed, and the fixed
split seed.

## Dataset roles

| Source | How AuditDDI uses it | Leakage and coverage rule |
|---|---|---|
| TWOSIDES | Main DDI positive labels and pair identities. Unreported pairs are sampled as negatives. | Keep unordered-pair duplicates together. A sampled negative means unreported, not known safe. S1 and S2 drug identities must be absent from training as specified by the split. |
| PubChem | Drug identity and structure cross-references used while resolving records to standardized structures. | PubChem is not a DDI label source and does not directly supply the model's Morgan bits; RDKit calculates those from standardized SMILES. Record failed and ambiguous mappings instead of guessing. |
| ChEMBL | Optional molecular encoder pretraining from molecular structures/assays. | Use a provenance-checked encoder-only checkpoint. Do not treat a generic model checkpoint as ChEMBL pretraining. For a strict cold-start comparison, pretraining must exclude evaluation drugs, as required by the ChEMBL pretraining protocol. |
| PharmGKB | Drug-to-gene/enzyme/pathway evidence, represented as drug-level features and pairwise overlap. | Preserve source IDs and matching evidence. Missing annotation must have a mask and must not be interpreted as a confirmed absence. |
| BindingDB | Drug-to-protein/target activity or affinity profiles and pairwise target overlap. | Keep assay units and evidence provenance during aggregation. Do not create target labels from DDI test outcomes. |
| UniProt | Protein sequences for targets linked to drugs by curated target evidence. | A sequence must be linked through a target accession or curated mapping; do not assign a generic sequence to an unmapped drug. |
| PDB | Drug–protein structure or target-complex evidence where a valid ligand/target mapping exists. | Coverage is expected to be sparse. A parsed target-presence vector is not the same as a learned 3D pocket embedding; report which representation the pipeline actually builds. |
| GEO | Drug perturbation expression profiles where compound, dose, cell/tissue, and experiment mappings are available. | Retain context and missingness. Do not merge incompatible experiments into a single unlabeled value without a documented aggregation rule. |
| FAERS | Individual-drug adverse-event/toxicity evidence for safety auditing and possible auxiliary learning. | Full-history scores may contain reports made after or because of the DDI events being predicted. The primary DDI run therefore excludes the FAERS score as an input feature. Use it in the DDI model only in a separately named, time-restricted ablation with report dates and label overlap audited. |

The multimodal model combines molecular graphs with available PharmGKB,
BindingDB/UniProt, GEO, and PDB features. ChEMBL is an encoder initialization,
not a second DDI label table. PubChem resolves identity/structure; it is not
counted as an independent predictive modality. FAERS is loaded and its coverage
is recorded, but its full-history score is disabled as a primary DDI input.

## Training and test boundaries

1. Resolve each drug to one canonical structure and stable identifiers before
   joining source tables. The split builder now maps source identifiers to the
   exact master `drug_id` used by the molecular cache, drops ambiguous aliases,
   and records unmatched counts. Save unmatched rows for review.
2. Build one enriched master-node snapshot. Each node contains the features
   available for that drug plus explicit source/missingness masks.
3. Build the TWOSIDES pair labels, remove conflicting or duplicate unordered
   pairs, then create split-aware negatives. Save exact split CSVs and hashes.
   The Colab launcher requires at least 50 positives and 50 sampled negatives
   in every train, development, and test partition, and stops before training
   if any split is empty or undersized. Do not lower this guard to force a run.
4. Fit the DDI model on training pairs only. Select the checkpoint using the
   mean AUROC across transductive validation, S1-dev, and S2-dev. Select each
   split's classification threshold using its matching validation/dev data.
   Evaluate transductive, S1, and S2 test sets once after model selection.
5. Keep a separate scaffold-disjoint study and independent external dataset
   evaluation. Do not choose a model or threshold from those test results.
6. Compare each data source by ablation on the same split hashes. A full
   multimodal result is useful only if it improves held-out results over the
   chemical baseline and retains usable coverage on S1/S2.

The multimodal trainer has been changed so the epoch loop reads only
transductive validation and separate S1/S2 development splits, never S1/S2
test metrics. The checkpoint uses the mean AUROC over those development splits;
thresholds are frozen from their matching development split before final test
evaluation. Optional molecular
self-supervised warm-up and pseudo-pair generation are limited to training
drug identities. It writes a per-seed input manifest with the master-node hash,
split hashes, feature dimensions, modality coverage, and seeds.

ChEMBL pretraining is run as a separate candidate through the audited
`src/training/run_experiment_suite.py` protocol because its current checkpoint
split contract does not match this multimodal benchmark. It must not be silently
loaded into this candidate.

## One seed per Colab account/session

Use the same shared writable Drive output folder and the same fixed split seed
for every run. Change only the model seed. The runner requires a Colab GPU,
keeps a source snapshot, reuses the same split directory, and writes separate
`seed_<number>` folders. A completed seed is left untouched if the command is
run again.

Start a fresh study under a new v2 folder so the invalid tiny split files and
completed seeds from the earlier run are not reused. First run the preflight;
it chooses the TWOSIDES file before a prefiltered unified edge file and prints
the class counts for each partition:

```bash
!python src/training/run_multimodal_seed.py --prepare-only --seed 11 --split-seed 42 --epochs 200 --holdout-fraction 0.30 --minimum-per-class 50 --output-dir /content/drive/MyDrive/auditddi-results/multimodal_seed_study_v2
```

If the file resolver chooses the wrong edge CSV, pass the full raw TWOSIDES
path with `--edges` and the enriched node CSV with `--master-nodes`. Check the
preflight's resolved edge path and split counts before training.

In Colab, mount the Drive folder that contains the repository and datasets,
select a GPU runtime, then run one seed:

```bash
!python src/training/run_multimodal_seed.py --seed 11 --split-seed 42 --epochs 200 --holdout-fraction 0.30 --minimum-per-class 50 --output-dir /content/drive/MyDrive/auditddi-results/multimodal_seed_study_v2
```

Run the other seeds in separate sessions/accounts, keeping `--split-seed 42`
and the shared output folder fixed:

```bash
!python src/training/run_multimodal_seed.py --seed 23 --split-seed 42 --epochs 200 --holdout-fraction 0.30 --minimum-per-class 50 --output-dir /content/drive/MyDrive/auditddi-results/multimodal_seed_study_v2
!python src/training/run_multimodal_seed.py --seed 37 --split-seed 42 --epochs 200 --holdout-fraction 0.30 --minimum-per-class 50 --output-dir /content/drive/MyDrive/auditddi-results/multimodal_seed_study_v2
!python src/training/run_multimodal_seed.py --seed 53 --split-seed 42 --epochs 200 --holdout-fraction 0.30 --minimum-per-class 50 --output-dir /content/drive/MyDrive/auditddi-results/multimodal_seed_study_v2
!python src/training/run_multimodal_seed.py --seed 71 --split-seed 42 --epochs 200 --holdout-fraction 0.30 --minimum-per-class 50 --output-dir /content/drive/MyDrive/auditddi-results/multimodal_seed_study_v2
```

After all selected seed folders are present in the shared Drive folder, aggregate
them only after the script confirms matching master-node and split hashes:

```bash
!python src/training/summarize_multimodal_seeds.py --study-dir /content/drive/MyDrive/auditddi-results/multimodal_seed_study_v2 --seeds 11 23 37 53 71
```

The summary writes seed-level metrics to `multimodal_seed_metrics.csv` and
descriptive means and bootstrap intervals to `multimodal_seed_summary.json`.

By default, the runner searches the resolved `AUDITDDI_DATA_BASE` and
`AUDITDDI_RESULTS_BASE` locations for the enriched master-node table and DDI
edges. If your Drive layout differs, pass `--master-nodes`, `--edges`,
`--splits-dir`, and `--output-dir` with their mounted Drive paths. Keep the
output and split locations shared and writable across accounts; if that is not
possible, copy each complete `seed_<number>` folder to one results folder after
training. Do not combine runs with different split hashes.

## What to compare

For each seed, retain AUROC and AUPRC for ranking, MCC and balanced accuracy
for thresholded classification, sensitivity/recall and specificity at the
frozen validation threshold, and Brier/ECE for calibration. Report seed-level
values and mean/intervals; do not report only the best seed. Accuracy is
secondary because the negative class is sampled unreported pairs and its
meaning depends on the sampling protocol.

The code change and protocol make the experiment reproducible, but they do not
promise an S1/S2 gain. That can only be established after the Drive-backed runs
complete and are compared against the same-split chemical baseline.
