"""Names for the drugs DrugCentral could not name (and a second name per drug), from PubChem.

    python -m src.features.pubchem_names --table .../event_table_v1 --out .../pubchem_v1

One PubChem request per drug InChIKey (title + up to 30 short synonyms), cached on disk, resumable.
A drug PubChem does not know stays unnamed; nothing is guessed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd


def rdkit_inchikey(smiles: str) -> Optional[str]:
    from rdkit import Chem
    from rdkit.Chem import inchi

    m = Chem.MolFromSmiles(str(smiles))
    return inchi.MolToInchiKey(m) if m is not None else None


def _get_json(session, url, tries=4, pause=0.3):
    for i in range(tries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
        except Exception:
            pass
        time.sleep(pause * (i + 1) * 2)
    return None


def fetch_names(drugs: pd.DataFrame, session, cache_path: Path, inchikey_fn: Callable[[str], Optional[str]] = rdkit_inchikey, pause: float = 0.25) -> pd.DataFrame:
    cache = json.load(open(cache_path)) if Path(cache_path).exists() else {}
    for n, (idx, smi) in enumerate(zip(drugs.drug_idx, drugs.smiles)):
        key = inchikey_fn(smi)
        if not key or str(idx) in cache:
            continue
        j = _get_json(session, f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{key}/property/Title/JSON")
        rec = {"inchikey": key, "cid": None, "title": None, "synonyms": []}
        if j:
            props = j.get("PropertyTable", {}).get("Properties", [])
            if props:
                rec["cid"], rec["title"] = props[0].get("CID"), props[0].get("Title")
                s = _get_json(session, f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{rec['cid']}/synonyms/JSON")
                if s:
                    syn = s.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
                    rec["synonyms"] = [x for x in syn if len(x) <= 40][:30]
        if j is not None:
            cache[str(idx)] = rec
        if n % 40 == 0:
            json.dump(cache, open(cache_path, "w"))
        time.sleep(pause)
    json.dump(cache, open(cache_path, "w"))
    rows = [{"drug_idx": int(k), **{x: v[x] for x in ("inchikey", "cid", "title")}, "synonyms": "|".join(v["synonyms"])} for k, v in cache.items()]
    return pd.DataFrame(rows, columns=["drug_idx", "inchikey", "cid", "title", "synonyms"]).sort_values("drug_idx").reset_index(drop=True)


def main(argv: Optional[Sequence[str]] = None, session=None, inchikey_fn=rdkit_inchikey) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(list(argv) if argv is not None else None)
    if session is None:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = "AuditDDI-pubchem/1.0"
    a.out.mkdir(parents=True, exist_ok=True)
    df = fetch_names(pd.read_csv(a.table / "drugs.csv"), session, a.out / "pubchem_cache.json", inchikey_fn)
    df.to_csv(a.out / "drug_names_pubchem.csv", index=False)
    print({"drugs": int(len(df)), "with_title": int(df.title.notna().sum())})


if __name__ == "__main__":
    main()
