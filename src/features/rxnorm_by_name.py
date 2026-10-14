"""Re-query RxNorm / RxClass BY DRUG NAME to get ATC classes for the 645 drugs.

    python -m src.features.rxnorm_by_name \
        --names /content/drive/MyDrive/auditddi-data/p1_targets/drug_names_drugcentral.csv \
        --pubchem-names /content/drive/MyDrive/auditddi-data/dataset_audit/drug_names_pubchem.csv \
        --out /content/drive/MyDrive/auditddi-data/rxnorm_v2

WHY: the first RxNorm download sent SMILES strings; RxNorm only understands names, so every answer
was empty.  Names come from DrugCentral (INN) and, if given, PubChem titles.

Matching policy (no guessing): a drug gets an RxCUI only from an EXACT name match.  Approximate
matching is off unless --approximate is given, and approximate hits are labelled in match_type.
ATC codes are taken from every RxClass answer by pattern (a code like N02BA01), so the parser does
not depend on the exact JSON nesting.

Outputs: rxnorm_by_name.csv (drug_idx, name_used, rxcui, match_type, atc_codes)
         drug_atc.csv       (drug_idx, atc_code)  <- input for build_feature_store --atc
         rxnorm_report.json
ATC classes exist only for drugs that are already marketed: features built from them belong to the
ANNOTATION-RICH scenario, never to the structure-only (true cold-start) scenario.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

BASE = "https://rxnav.nlm.nih.gov/REST"
ATC_CODE = re.compile(r"^[A-Z]\d{2}[A-Z]{0,2}\d{0,2}$")


def _get(session, url, params=None, tries: int = 3, pause: float = 0.2):
    for i in range(tries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
        except Exception:
            pass
        time.sleep(pause * (i + 1) * 2)
    return None


def find_atc_codes(obj) -> List[str]:
    """All ATC-looking values under any key called classId, anywhere in the JSON."""
    found: List[str] = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "classId" and isinstance(v, str) and ATC_CODE.match(v.strip().upper()):
                    found.append(v.strip().upper())
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return sorted(set(found))


def rxcui_exact(session, name: str) -> Optional[str]:
    j = _get(session, f"{BASE}/rxcui.json", {"name": name})
    ids = (j or {}).get("idGroup", {}).get("rxnormId") or []
    return str(ids[0]) if ids else None


def rxcui_approx(session, name: str) -> Optional[str]:
    j = _get(session, f"{BASE}/approximateTerm.json", {"term": name, "maxEntries": 1})
    cands = (j or {}).get("approximateGroup", {}).get("candidate") or []
    return str(cands[0].get("rxcui")) if cands and cands[0].get("rxcui") else None


def atc_for_rxcui(session, rxcui: str) -> List[str]:
    j = _get(session, f"{BASE}/rxclass/class/byRxcui.json", {"rxcui": rxcui, "relaSource": "ATC"})
    return find_atc_codes(j or {})


def resolve(session, drug_names: pd.DataFrame, approximate: bool = False, pause: float = 0.15, cache: Optional[dict] = None) -> pd.DataFrame:
    """drug_names: columns drug_idx, name (one row per candidate name, in priority order)."""
    cache = {} if cache is None else cache
    rows = []
    for idx, g in drug_names.groupby("drug_idx", sort=True):
        rec = {"drug_idx": int(idx), "name_used": None, "rxcui": None, "match_type": "none", "atc_codes": ""}
        for name in g["name"]:
            nm = str(name).strip().lower()
            if not nm or nm == "nan":
                continue
            if nm not in cache:
                rx, mt = rxcui_exact(session, nm), "exact"
                if rx is None and approximate:
                    rx, mt = rxcui_approx(session, nm), "approximate"
                cache[nm] = {"rxcui": rx, "match_type": mt if rx else "none", "atc": atc_for_rxcui(session, rx) if rx else []}
                time.sleep(pause)
            c = cache[nm]
            if c["rxcui"]:
                rec.update({"name_used": nm, "rxcui": c["rxcui"], "match_type": c["match_type"], "atc_codes": "|".join(c["atc"])})
                break
        rows.append(rec)
    return pd.DataFrame(rows)


def names_table(dc_names: pd.DataFrame, pubchem_names: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    parts = []
    if "inn" in dc_names:
        parts.append(dc_names[["drug_idx", "inn"]].rename(columns={"inn": "name"}).assign(prio=0))
    if pubchem_names is not None and "title" in pubchem_names:
        parts.append(pubchem_names[["drug_idx", "title"]].rename(columns={"title": "name"}).assign(prio=1))
    t = pd.concat(parts, ignore_index=True).dropna(subset=["name"])
    return t.sort_values(["drug_idx", "prio"]).drop_duplicates(["drug_idx", "name"])[["drug_idx", "name"]]


def to_long_atc(res: pd.DataFrame) -> pd.DataFrame:
    rows = [(int(r.drug_idx), c) for r in res.itertuples() for c in str(r.atc_codes).split("|") if c and c != "nan"]
    return pd.DataFrame(rows, columns=["drug_idx", "atc_code"]).drop_duplicates()


def main(argv: Optional[Sequence[str]] = None, session=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True, type=Path)
    ap.add_argument("--pubchem-names", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--approximate", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if session is None:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = "AuditDDI-rxnorm-by-name/1.0"
    a.out.mkdir(parents=True, exist_ok=True)
    cache_f = a.out / "rxnorm_name_cache.json"
    cache = json.load(open(cache_f)) if cache_f.exists() else {}
    nt = names_table(pd.read_csv(a.names), pd.read_csv(a.pubchem_names) if a.pubchem_names and a.pubchem_names.exists() else None)
    res = resolve(session, nt, a.approximate, cache=cache)
    json.dump(cache, open(cache_f, "w"))
    res.to_csv(a.out / "rxnorm_by_name.csv", index=False)
    atc = to_long_atc(res)
    atc.to_csv(a.out / "drug_atc.csv", index=False)
    rep = {"drugs": int(len(res)), "with_rxcui": int(res.rxcui.notna().sum()), "with_atc": int(atc.drug_idx.nunique()),
           "match_types": res.match_type.value_counts().to_dict(), "names_tried": int(len(nt))}
    (a.out / "rxnorm_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
