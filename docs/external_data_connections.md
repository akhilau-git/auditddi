# DDInter and DrugBank data connections

These connectors keep independent reference evidence separate from AuditDDI's model predictions. They do not alter the training labels or treat an absent database entry as a negative interaction.

## DDInter 2.0 exports

1. In the Google Drive folder `auditddi-data`, create `external/DDInter/raw`.
2. In Colab, after mounting Drive and adding the project to `sys.path`, download the eight files linked by the official [DDInter download page](https://ddinter2.scbdd.com/download/) and import them:

```python
from src.data_prep.ddinter_pipeline import download_official_exports, import_directory

data_root = "/content/drive/MyDrive/auditddi-data"
results_root = "/content/drive/MyDrive/auditddi-results"
download_official_exports(f"{data_root}/external/DDInter/raw")
manifest = import_directory(
    f"{data_root}/external/DDInter/raw",
    f"{results_root}/external/DDInter",
)
manifest
```

The exports cover ATC groups A, B, D, H, L, P, R, and V; this is not full coverage of all drug categories.

The importer preserves raw rows, normalizes pair IDs/names/severity/description where recognized, and writes a hash-bearing manifest. If the official export uses a changed schema, it stops with the observed column list; extend the aliases only after inspecting the downloaded header. DDInter is licensed CC BY-NC-SA 4.0; review the [official terms](https://ddinter2.scbdd.com/terms/) before use or redistribution.

The resulting table is reference evidence, not directly a binary benchmark. To evaluate the molecular model, first map DDInter identifiers/names to structures using a reviewed identifier crosswalk (retain mapping evidence and ambiguity), screen exact unordered pairs against all development splits, and define a defensible negative/control set. DDInter's unlisted pairs must remain unknown. Until these steps are complete, report coverage and known-positive retrieval/ranking only; do not report AUROC or specificity from unlabeled absences.

## DrugBank DDI API

The DrugBank tutorial also documents a region-scoped clinical DDI lookup: `GET /v1/{region}/ddi?product_concept_id=...`. This is distinct from the scientific API's POST endpoint accepting DrugBank IDs. The connector supports both. For the clinical endpoint it expects product-concept IDs (`DBPC...`) and a short-lived token in `AUDITDDI_DRUGBANK_TOKEN`; for the scientific endpoint it expects DrugBank IDs (`DB...`) and an API key in `AUDITDDI_DRUGBANK_API_KEY`. These credentials are not interchangeable. Obtain access appropriate to your intended use and add it to **Colab Secrets**; never put credentials in the frontend, notebook source, Git, or a message.

**License check before project use:** An API key proves authentication, not permission for every use or bulk extraction. DrugBank's current terms limit default access to internal, non-clinical educational/research use and restrict benchmarking/competitive analysis and creating competing DDI databases/software. Before using DrugBank results to train, benchmark, publish, or power an AuditDDI product, obtain written confirmation from DrugBank that the specific use and storage are permitted. The full academic dataset downloads are currently listed as temporarily paused. See [DrugBank Terms of Use](https://trust.drugbank.com/drugbank-trust-center/terms-of-use) and [current release/download status](https://go.drugbank.com/releases/latest).

```python
from src.data_prep.drugbank_api import lookup_product_concept_interactions, save_lookup

# In Colab, load the secret without printing it:
from google.colab import userdata
import os
os.environ["AUDITDDI_DRUGBANK_TOKEN"] = userdata.get("AUDITDDI_DRUGBANK_TOKEN")

ids = ["DBPC0180857", "DBPC0021865"]  # the product-concept pair from the DrugBank tutorial
response = lookup_product_concept_interactions(ids, region="us")
saved_path = save_lookup(
    response,
    ids,
    "/content/drive/MyDrive/auditddi-results/external/DrugBank",
)
print(saved_path)
```

The client sends `Authorization: Bearer <token>` as described in DrugBank's token-authentication documentation. The API response is archived with query IDs and retrieval time. This is a source lookup, not independent model validation by itself. For the dashboard, call it only from the backend after server-side authentication/authorization; never call it from browser JavaScript with the secret key. The local connector currently provides research ingestion only; it does not change model predictions or add a public API route.

## Current boundary

The project can now ingest DDInter exports and make authorized DrugBank DDI API queries. This workspace has no access to the user's Drive mount or DrugBank subscription key, so downloading their files, confirming the live API response/schema, and producing validated external metrics must happen after the owner mounts Drive and supplies the key through Colab Secrets. API access to DrugBank and the academic data-download license are separate products/permissions.
