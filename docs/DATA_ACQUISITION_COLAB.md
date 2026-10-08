# Dataset acquisition in Google Colab

The implementation guide lists these source families: RxNorm, PubChem,
TWOSIDES, DDInter, KEGG, openFDA, DailyMed, DrugCentral, ChEMBL, BindingDB,
PharmGKB, UniProt, PDB, and FAERS.  They are kept in separate folders because
source evidence is not interchangeable and an unreported interaction is not a
confirmed negative.

The acquisition script creates the project-local layout:

```text
AuditDDI/
└── auditddi-data/
    ├── bindingdb/
    ├── chembl/
    ├── dailymed/
    ├── ddinter/
    ├── drugcentral/
    ├── faers/
    ├── kegg/
    ├── openfda/
    ├── pdb/
    ├── pharmgkb/
    ├── pubchem/
    ├── rxnorm/
    ├── twosides/
    ├── uniprot/
    └── collection_manifest.json
```

## Colab command

Clone the repository into the same Drive project folder, then run the
collector. Replace each example URL with the official release URL or API
export that you are authorized to use. The script never guesses a release URL
and never silently creates a fake or empty dataset.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content/drive/MyDrive/AuditDDI
python scripts/collect_datasets_colab.py \
  --project-root /content/drive/MyDrive/AuditDDI \
  --source all \
  --url twosides=REPLACE_WITH_OFFICIAL_TWOSIDES_URL \
  --url faers=REPLACE_WITH_AUTHORIZED_FAERS_EXPORT_URL \
  --url chembl=REPLACE_WITH_CHEMBL_RELEASE_URL \
  --allow-licensed
```

Add one `--url SOURCE=URL` argument for every source release to be downloaded.
Only include `--allow-licensed` when the listed restricted sources are covered
by written authorization; otherwise omit those `--url` arguments.
For sources retrieved through an API, first export the authorized response to
a stable URL/file and pass that snapshot URL; this preserves a reproducible
input instead of fetching an unversioned API during training.

`--allow-licensed` is required for KEGG, DrugCentral, ChEMBL, BindingDB, and
PharmGKB. The command refuses DrugBank by default and does not support
Medi-Span. Use either only after written permission covers the intended
research, storage, and redistribution. Never put API keys in the command or
repository.

Every downloaded file is written atomically under its source folder. The
resulting `auditddi-data/collection_manifest.json` records the exact URL,
retrieval time, byte count, and SHA-256 digest. A source with no URL is marked
`not_requested`; it must not be described as collected. The manifest is
evidence of acquisition only, not proof that a source was used by a training
run.

## Verify datasets that are already present

Do not download sources that already exist in Drive. Run the read-only
verifier against the actual shared data root:

```python
!python /content/drive/MyDrive/AuditDDI/scripts/verify_existing_datasets.py \
  --data-root /content/drive/MyDrive/auditddi-data
```

This writes:

```text
/content/drive/MyDrive/auditddi-data/verification_manifest.json
```

The manifest records every file's size and SHA-256, checks required TWOSIDES
and DDInter files, and records missing folders as `needs_review`. It does not
change source files and it does not treat a folder's existence as proof of
license, release identity, or training use. DrugBank and Medi-Span remain
explicitly excluded.

## Collect RxNorm, openFDA, and DailyMed evidence

After verifying the existing sources, use the TWOSIDES drug catalog as the
query list. This performs cached public API retrieval; it does not download
TWOSIDES again and does not turn regulatory evidence into DDI labels.

```python
!python /content/drive/MyDrive/AuditDDI/scripts/collect_public_evidence_colab.py \
  --data-root /content/drive/.shortcut-targets-by-id/1EK5SEg3iwEAEUBzwrCOsj_Y0huxGZklA/auditddi-data \
  --drug-file /content/drive/.shortcut-targets-by-id/1EK5SEg3iwEAEUBzwrCOsj_Y0huxGZklA/auditddi-data/twosides/twosides_drugs.csv \
  --drug-column drug_id \
  --source rxnorm \
  --source openfda \
  --source dailymed \
  --delay 0.5
```

The command writes raw JSON responses and a manifest under:

```text
auditddi-data/rxnorm/
auditddi-data/openfda/
auditddi-data/dailymed/
```

Each source manifest records the queried drug name, exact URL, retrieval time,
HTTP result, byte count, and SHA-256. HTTP errors are retained and marked
rather than silently treated as successful matches. These API queries provide
identity or regulatory evidence only; they are not pairwise DDI truth.
