"""Server-side DrugBank DDI API client; credentials are read only from env."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests


DDI_URL = "https://api.drugbank.com/v1/ddi"
CLINICAL_DDI_BASE_URL = "https://api.drugbank.com/v1"


def lookup_product_concept_interactions(
    product_concept_ids: list[str],
    *,
    region: str = "us",
    token: str | None = None,
    timeout: int = 30,
) -> requests.Response:
    """Use the region-scoped clinical DDI endpoint shown in DrugBank's tutorial."""
    ids = list(dict.fromkeys(value.strip() for value in product_concept_ids if value.strip()))
    if len(ids) < 2:
        raise ValueError("Provide at least two DrugBank product-concept IDs (DBPC...).")
    if any(not value.upper().startswith("DBPC") for value in ids):
        raise ValueError("This endpoint expects product-concept IDs such as DBPC0180857, not DB drug IDs.")
    if not region.isalpha() or len(region) != 2:
        raise ValueError("Region must be a two-letter code such as 'us'.")
    credential = token or os.environ.get("AUDITDDI_DRUGBANK_TOKEN")
    if not credential:
        raise RuntimeError("Set AUDITDDI_DRUGBANK_TOKEN in the environment/Colab Secret. Do not put it in source or frontend code.")
    response = requests.get(
        f"{CLINICAL_DDI_BASE_URL}/{region.lower()}/ddi",
        headers={"Authorization": f"Bearer {credential}", "Accept": "application/json"},
        params={"product_concept_id": ",".join(ids)},
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def lookup_interactions(drugbank_ids: list[str], *, api_key: str | None = None, timeout: int = 30) -> requests.Response:
    ids = list(dict.fromkeys(value.strip() for value in drugbank_ids if value.strip()))
    if len(ids) < 2:
        raise ValueError("Provide at least two DrugBank IDs (for example, DB00503 and DB01221).")
    if any(not value.upper().startswith("DB") for value in ids):
        raise ValueError("Only DrugBank IDs are accepted; resolve names to IDs through an authorized workflow first.")
    credential = api_key or os.environ.get("AUDITDDI_DRUGBANK_API_KEY")
    if not credential:
        raise RuntimeError("Set AUDITDDI_DRUGBANK_API_KEY in the environment/Colab Secret. Never put it in source or frontend code.")
    response = requests.post(
        DDI_URL,
        headers={"Authorization": credential, "Accept": "application/json", "Content-Type": "application/json"},
        json={"drugbank_id": ids},
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def save_lookup(response: requests.Response, query_ids: list[str], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset_name": "DrugBank DDI API",
        "source_url_or_doi": DDI_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_identifiers": query_ids,
        "label_definition": "DrugBank API interaction result; non-returned pairs are unknown, not negative.",
        "response": response.json(),
    }
    path = destination / f"drugbank_ddi_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("drugbank_ids", nargs="+", help="DrugBank identifiers, e.g. DB00503 DB01221")
    parser.add_argument("--output-dir", type=Path, default=Path("results/external/DrugBank"))
    args = parser.parse_args()
    response = lookup_interactions(args.drugbank_ids)
    print(f"Saved DrugBank DDI evidence to {save_lookup(response, args.drugbank_ids, args.output_dir)}")


if __name__ == "__main__":
    main()
