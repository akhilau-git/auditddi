# AuditDDI seven-branch model

`auditddi_seven_branch_v1` implements the seven per-drug inputs in the proposed
system:

1. PubChem SMILES → molecular graph → shared edge-aware GATv2 encoder.
2. PubChem SMILES → 1024-bit Morgan fingerprint → shared fingerprint MLP.
3. Drug profile → MLP with an explicit availability mask.
4. Adverse-event profile → MLP with an explicit availability mask.
5. Drug–protein target profile → MLP with an explicit availability mask.
6. Drug-linked 3D structure profile → MLP with an explicit availability mask.
7. Context-defined gene-expression profile → MLP with an explicit availability mask.

For each drug, the seven encoded representations are concatenated. The pair
head follows the proposed `[hA || hB || hA - hB]` fusion. Because the task is an
undirected binary DDI association, the shared MLP scores both drug orders and
averages their logits. This preserves the proposed fusion while guaranteeing
that swapping drug order cannot change the prediction. The old checkpoint
architecture remains separate; a checkpoint trained with it cannot be resumed
as this model.

## Compact feature-table contract

The molecular graph and Morgan fingerprint are generated from the existing
PubChem SMILES during training. The other five branches are supplied through
one compact `.npz` file. Prepare one CSV per branch, each with:

- `canonical_smiles` as the first identity key;
- one numeric column per feature;
- at most one row per canonical drug. Aggregate GEO conditions or multiple
  protein/structure observations upstream using an explicit documented rule.

The backward-compatible `processed/auditddi_v2/source_feature_matrix.npz`
still has 51 columns, but it is not the required input for the new model. Build
a variable-width `drug_profile.csv` from the full mapped PharmGKB drug-gene
annotation table. It defaults to up to 2,048 observed genes (with no minimum
frequency cutoff), keyed through the selected PubChem CID crosswalk. The
profile builder records its selected vocabulary and coverage. Missing drugs
are omitted so the bundle builder can represent them with the explicit branch
mask. This currently implements the real mapped **gene-annotation** portion
of the drug profile; it does not claim measured gene expression, variant
features, or pathway features. Those require their own source evidence.

Run after `prepare_auditddi_source_bundle.py` has emitted the mapped long table:

```bash
python scripts/build_pharmacogenomic_drug_profile.py \
  --data-root /content/drive/MyDrive/Projects/auditddi
```

It writes `processed/seven_branch/drug_profile.csv` and
`drug_profile.manifest.json`. The openFDA adverse-reaction extractor streams
the label ZIP files and deliberately ignores pairwise `drug_interactions`
text:

```bash
python scripts/build_openfda_adverse_event_features.py \
  --data-root /content/drive/MyDrive/Projects/auditddi \
  --output-dir /content/drive/MyDrive/Projects/auditddi/processed/seven_branch
```

It writes `adverse_event.csv` and `adverse_event.manifest.json`. The CSV is a
compact, drug-keyed TF-IDF representation of the label `adverse_reactions`
section; it is not an incidence estimate. Review mapped-drug coverage and
label-date/source overlap before training. The other required tables are
`protein_target.csv`,
`pdb_structure.csv`, and `gene_expression.csv`. Missing drugs are represented
by an explicit row mask in the bundle. The bundle builder rejects partially
missing feature rows and duplicate drug rows. It does not fabricate branch
data or infer missing values silently.

### ChEMBL protein/target preparation

The existing ChEMBL 37 chemreps file supplies structures, not drug–target
measurements. The extractor below exact-matches canonical SMILES to ChEMBL
molecule IDs, then uses the official paginated activity endpoint. It filters to
human records with confidence score ≥8, binding or functional assays, exact
standard relations, and pChEMBL-backed IC50/Ki/Kd/EC50/AC50 measurements. Assay
confidence is fetched from the ChEMBL assay endpoint and joined to activity
records by assay ChEMBL ID (the activity response does not reliably include
that field). Activity replicates are collapsed by median per drug–target pair; the most widely
observed targets are retained, with a deterministic target-ID tie break. Each
target contributes a standardized affinity value and an observed bit, so
missing evidence is distinguishable from a measured value. The script caches
complete API batches in Drive and writes its filters, API release, hashes, and
coverage to a manifest.

Run in Colab after Drive is mounted and the updated project source is available:

```bash
python scripts/build_chembl_protein_target_features.py \
  --data-root /content/drive/MyDrive/Projects/auditddi \
  --chemreps /content/drive/MyDrive/Projects/auditddi/chembl/chembl_37_chemreps.txt.gz \
  --output-dir /content/drive/MyDrive/Projects/auditddi/processed/seven_branch
```

Review `protein_target.manifest.json` and `protein_target.csv` before bundling.
The branch represents measured human drug–target activity profiles; target
sequence embeddings are a separate enhancement and must not be claimed from
these affinity features alone. The ChEMBL API returns paginated results and
supports field filters and selected fields, as described in its
[official data web-service documentation](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services).

After the source splits and feature tables are present in Drive, audit branch
coverage on every saved split before training:

```bash
python scripts/audit_seven_branch_split_coverage.py \
  --data-root /content/drive/MyDrive/Projects/auditddi
```

The report is saved as `processed/seven_branch/split_branch_coverage.csv` and
includes drug coverage plus pair coverage where both endpoints, or at least
one endpoint, has an observed feature row. This audit does not modify the
splits or impute missing branch data.

Also provide a provenance JSON with a `branches` object. Each branch must name
its `dataset`, `release`, `feature_definition`, and `drug_identity_mapping`.
Adverse-event provenance must also include `pairwise_ddi_signals_excluded: true`
and a `cutoff_date`; protein-target provenance must state `activity_encoding`;
PDB provenance must state `structure_to_drug_mapping`; GEO provenance must
state `cell_or_tissue_context`, `perturbation_type`, and `aggregation_rule`.
This records choices needed to audit leakage and biological context.

Example, after placing these compact tables under the mounted Drive data root:

```bash
python scripts/build_seven_branch_feature_bundle.py \
  --drug-profile /content/drive/MyDrive/Projects/auditddi/processed/seven_branch/drug_profile.csv \
  --adverse-event /content/drive/MyDrive/Projects/auditddi/processed/seven_branch/adverse_event.csv \
  --protein-target /content/drive/MyDrive/Projects/auditddi/processed/seven_branch/protein_target.csv \
  --pdb-structure /content/drive/MyDrive/Projects/auditddi/processed/seven_branch/pdb_structure.csv \
  --gene-expression /content/drive/MyDrive/Projects/auditddi/processed/seven_branch/gene_expression.csv \
  --provenance-json /content/drive/MyDrive/Projects/auditddi/processed/seven_branch/provenance.json \
  --output /content/drive/MyDrive/Projects/auditddi/processed/auditddi_v3/seven_branch_features.npz
```

The output includes feature names, source file hashes, row coverage, and
availability masks. The trainer checks the bundle schema, hashes, dimensions,
finite values, and non-zero observed coverage for every branch before fitting.

## Training configuration

Use a new run and a ChEMBL-pretrained graph encoder. Do not point the new
architecture at an old three-input checkpoint. For a 20-epoch pilot, configure
the Colab process before launching the trainer:

```python
import os
os.environ['AUDITDDI_MODEL_ARCHITECTURE'] = 'auditddi_seven_branch_v1'
os.environ['AUDITDDI_SEVEN_BRANCH_FEATURE_BUNDLE'] = (
    '/content/drive/MyDrive/Projects/auditddi/processed/auditddi_v3/'
    'seven_branch_features.npz'
)
os.environ['AUDITDDI_PRETRAINED_ENCODER_PATH'] = (
    '/content/drive/MyDrive/Projects/auditddi/checkpoints/'
    'chembl_pretrained_encoder.pt'
)
os.environ['AUDITDDI_EPOCHS'] = '20'
os.environ['AUDITDDI_EARLY_STOPPING_MIN_EPOCHS'] = '20'
```

The run still uses TWOSIDES as the supervised pair-label source and excludes
known DDInter pairs from sampled negatives. Sampled negatives mean
“unreported/unknown,” not “safe.” Before using FAERS adverse-event features,
audit their reporting window and remove pairwise interaction-derived signals
that could overlap the TWOSIDES labels. For GEO, record perturbagen, cell/tissue,
dose, and duration, then document the rule used to aggregate contexts to one
drug vector. Good performance cannot be guaranteed by architecture alone; judge
the pilot using held-out metrics, both loss curves, coverage audits, and the
existing split protocol before deciding on a 200-epoch run.
