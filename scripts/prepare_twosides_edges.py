"""Prepare the TWOSIDES name table as a canonical-SMILES edge table.

The resolver is deliberately fail-closed: an unresolved name is reported and
its rows are omitted rather than being assigned a guessed structure.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from src.data_prep.pubchem_bridge import canonicalize


PUBCHEM_NAME_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/"
    "CanonicalSMILES/TXT"
)
PUBCHEM_RXNORM_CIDS_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RN/{}/cids/TXT"
)
PUBCHEM_CID_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/"
    "CanonicalSMILES/TXT"
)


@dataclass(frozen=True)
class Lookup:
    smiles: str | None
    status: str
    attempts: int
    query_url: str


def _request_text(
    url: str,
    *,
    request_get=requests.get,
    retries: int = 3,
    timeout: int = 30,
    sleep=time.sleep,
) -> tuple[str | None, int, str]:
    """Return response text with bounded retry handling and provenance."""
    for attempt in range(1, retries + 1):
        try:
            response = request_get(url, timeout=timeout)
        except requests.RequestException:
            if attempt == retries:
                return None, attempt, "network_error"
            sleep(attempt)
            continue
        if response.status_code == 200:
            text = response.text.strip()
            return (text or None), attempt, ("matched" if text else "empty_response")
        if response.status_code == 404:
            return None, attempt, "not_found_http_404"
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < retries:
                sleep(attempt)
                continue
        return None, attempt, f"http_{response.status_code}"
    raise AssertionError("retry loop exited unexpectedly")


def lookup_pubchem(
    name: str,
    rxnorm_id: str | int | float | None = None,
    *,
    request_get=requests.get,
    retries: int = 3,
    timeout: int = 30,
    sleep=time.sleep,
) -> Lookup:
    """Resolve by name first, then by the TWOSIDES RxNorm identifier."""
    cleaned = str(name).strip()
    name_url = PUBCHEM_NAME_URL.format(quote(cleaned, safe=""))
    text, attempts, status = _request_text(
        name_url,
        request_get=request_get,
        retries=retries,
        timeout=timeout,
        sleep=sleep,
    )
    if text:
        smiles = canonicalize(text)
        if smiles:
            return Lookup(smiles, "matched_by_name", attempts, name_url)
        status = "invalid_smiles_by_name"

    rxnorm = None
    if rxnorm_id is not None and pd.notna(rxnorm_id):
        rxnorm = str(rxnorm_id).strip()
        if rxnorm.endswith(".0"):
            rxnorm = rxnorm[:-2]
    if rxnorm and rxnorm.lower() not in {"nan", "none", "null"}:
        xref_url = PUBCHEM_RXNORM_CIDS_URL.format(quote(rxnorm, safe=""))
        cids, xref_attempts, xref_status = _request_text(
            xref_url,
            request_get=request_get,
            retries=retries,
            timeout=timeout,
            sleep=sleep,
        )
        if cids:
            cid = cids.splitlines()[0].strip()
            cid_url = PUBCHEM_CID_URL.format(quote(cid, safe=""))
            smiles_text, cid_attempts, cid_status = _request_text(
                cid_url,
                request_get=request_get,
                retries=retries,
                timeout=timeout,
                sleep=sleep,
            )
            smiles = canonicalize(smiles_text)
            if smiles:
                return Lookup(
                    smiles,
                    "matched_by_rxnorm",
                    attempts + xref_attempts + cid_attempts,
                    cid_url,
                )
            status = f"rxnorm_{cid_status}"
        else:
            status = f"rxnorm_{xref_status}"
        attempts += xref_attempts
    return Lookup(None, status, attempts, name_url)


def _load_cache(path: Path) -> dict[str, Lookup]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: Lookup(**value) for key, value in payload.items()}


def _save_cache(path: Path, cache: dict[str, Lookup]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({key: asdict(value) for key, value in cache.items()}, indent=2),
        encoding="utf-8",
    )


def prepare_edges(
    raw: Path,
    output: Path,
    *,
    cache_path: Path | None = None,
    report_path: Path | None = None,
    chunksize: int = 100_000,
    delay: float = 0.2,
) -> dict[str, int]:
    """Resolve all unique TWOSIDES drugs and write the prepared edge file."""
    required = {
        "drug_1_concept_name",
        "drug_2_concept_name",
        "drug_1_rxnorm_id",
        "drug_2_rxnorm_id",
        "condition_concept_name",
    }
    names: dict[str, set[str]] = {}
    for chunk in pd.read_csv(raw, compression="gzip", usecols=required, chunksize=chunksize):
        for name_col, id_col in (
            ("drug_1_concept_name", "drug_1_rxnorm_id"),
            ("drug_2_concept_name", "drug_2_rxnorm_id"),
        ):
            values = chunk[[name_col, id_col]].dropna(subset=[name_col])
            for name, rxnorm in values.itertuples(index=False):
                names.setdefault(str(name).strip(), set()).add(str(rxnorm).strip())

    cache = _load_cache(cache_path) if cache_path else {}
    name_to_lookup: dict[str, Lookup] = {}
    for index, name in enumerate(sorted(names), start=1):
        cache_key = f"name:{name.upper()}"
        lookup = cache.get(cache_key)
        if lookup is None:
            rxnorm = sorted(names[name])[0] if names[name] else None
            lookup = lookup_pubchem(name, rxnorm)
            cache[cache_key] = lookup
            if delay:
                time.sleep(delay)
        name_to_lookup[name] = lookup
        if index % 25 == 0:
            print(f"{index} / {len(names)}")
            if cache_path:
                _save_cache(cache_path, cache)
    if cache_path:
        _save_cache(cache_path, cache)

    unresolved = [
        {"name": name, **asdict(lookup)}
        for name, lookup in name_to_lookup.items()
        if lookup.smiles is None
    ]
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(unresolved, indent=2), encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True
    written = skipped = 0
    for chunk in pd.read_csv(
        raw,
        compression="gzip",
        usecols=["drug_1_concept_name", "drug_2_concept_name", "condition_concept_name"],
        chunksize=chunksize,
    ):
        chunk["source"] = chunk["drug_1_concept_name"].map(
            lambda value: name_to_lookup.get(str(value).strip(), Lookup(None, "", 0, "")).smiles
        )
        chunk["target"] = chunk["drug_2_concept_name"].map(
            lambda value: name_to_lookup.get(str(value).strip(), Lookup(None, "", 0, "")).smiles
        )
        chunk["interaction_type"] = chunk["condition_concept_name"].fillna("").astype(str).str.strip()
        valid = chunk[
            chunk["source"].notna()
            & chunk["target"].notna()
            & chunk["interaction_type"].ne("")
        ][["source", "target", "interaction_type"]]
        skipped += len(chunk) - len(valid)
        written += len(valid)
        valid.to_csv(output, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
    return {
        "unique_names": len(names),
        "unresolved_names": len(unresolved),
        "rows_written": written,
        "rows_skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--unresolved-report", type=Path)
    args = parser.parse_args()
    print(prepare_edges(args.raw, args.output, cache_path=args.cache, report_path=args.unresolved_report))


if __name__ == "__main__":
    main()
