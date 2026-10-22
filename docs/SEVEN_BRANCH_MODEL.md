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
head uses `[hA + hB, abs(hA - hB), hA * hB]`, which makes the binary DDI score
invariant to drug order. The old checkpoint architecture remains separate; a
checkpoint trained with it cannot be resumed as this model.

## Compact feature-table contract

The molecular graph and Morgan fingerprint are generated from the existing
PubChem SMILES during training. The other five branches are supplied through
one compact `.npz` file. Prepare one CSV per branch, each with:

- `canonical_smiles` as the first identity key;
- one numeric column per feature;
- at most one row per canonical drug. Aggregate GEO conditions or multiple
  protein/structure observations upstream using an explicit documented rule.

Required tables: the existing `processed/auditddi_v2/source_feature_matrix.npz`
can be used directly for `drug_profile` (it has `canonical_smiles`, `features`,
and `feature_names`). It is the current 51-column PharmGKB/openFDA profile; its
reserved zero FAERS slot does **not** replace the adverse-event branch. The
other required tables are `adverse_event.csv`, `protein_target.csv`,
`pdb_structure.csv`, and `gene_expression.csv`. Missing drugs are represented
by an explicit row mask in the bundle. The bundle builder rejects partially
missing feature rows and duplicate drug rows. It does not fabricate branch
data or infer missing values silently.

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
  --drug-profile /content/drive/MyDrive/Projects/auditddi/processed/auditddi_v2/source_feature_matrix.npz \
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
