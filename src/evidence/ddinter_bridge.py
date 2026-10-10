"""Map DDInter (curated interactions with severity) onto your 645 drugs.

    python -m src.evidence.ddinter_bridge \
        --table /content/drive/MyDrive/auditddi-data/event_table_v1 \
        --ddinter /content/drive/MyDrive/auditddi-data/ddinter/raw \
        --names /content/drive/MyDrive/auditddi-data/p1_targets/drug_names_drugcentral.csv \
        --pubchem-names /content/drive/MyDrive/auditddi-data/dataset_audit/drug_names_pubchem.csv \
        --out /content/drive/MyDrive/auditddi-data/evidence_v1

Output ddinter_pairs.csv: drug_a < drug_b (event-table indices), severity (worst seen), n_records,
in_twosides (is the pair also a TWOSIDES pair?).  DDInter is EXTERNAL VALIDATION ONLY: it is pair-level
interaction knowledge, so using it as a training feature would leak the answer.
Only unambiguous names are matched; an unmatched DDInter drug is dropped, never guessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from src.evidence.openfda_labels import build_name_map, names_table, norm

SEVERITY_RANK = {"unknown": 0, "minor": 1, "moderate": 2, "major": 3}


def load_ddinter(raw_dir: Path) -> pd.DataFrame:
    parts = [pd.read_csv(f) for f in sorted(Path(raw_dir).glob("ddinter_downloads_code_*.csv"))]
    df = pd.concat(parts, ignore_index=True)
    df["a"] = df["Drug_A"].map(norm)
    df["b"] = df["Drug_B"].map(norm)
    df["level_norm"] = df["Level"].map(lambda x: norm(x) if isinstance(x, str) else "unknown")
    return df


def bridge(ddinter: pd.DataFrame, name_map: Dict[str, int], twosides_pairs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    ia, ib = ddinter["a"].map(name_map), ddinter["b"].map(name_map)
    ok = ia.notna() & ib.notna() & (ia != ib)
    d = ddinter[ok].copy()
    d["drug_a"] = pd.concat([ia[ok], ib[ok]], axis=1).min(axis=1).astype(int)
    d["drug_b"] = pd.concat([ia[ok], ib[ok]], axis=1).max(axis=1).astype(int)
    d["rank"] = d["level_norm"].map(lambda x: SEVERITY_RANK.get(x, 0))
    g = d.groupby(["drug_a", "drug_b"], as_index=False).agg(rank=("rank", "max"), n_records=("rank", "size"))
    inv = {v: k for k, v in SEVERITY_RANK.items()}
    g["severity"] = g["rank"].map(inv)
    if twosides_pairs is not None:
        two = set(zip(twosides_pairs[["drug_a", "drug_b"]].min(axis=1), twosides_pairs[["drug_a", "drug_b"]].max(axis=1)))
        g["in_twosides"] = [(int(a), int(b)) in two for a, b in zip(g.drug_a, g.drug_b)]
    return g.drop(columns="rank")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--ddinter", required=True, type=Path)
    ap.add_argument("--names", required=True, type=Path)
    ap.add_argument("--pubchem-names", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(list(argv) if argv is not None else None)
    pc = pd.read_csv(a.pubchem_names) if a.pubchem_names and a.pubchem_names.exists() else None
    nm, n_amb = build_name_map(names_table(pd.read_csv(a.names), pc))
    dd = load_ddinter(a.ddinter)
    g = bridge(dd, nm, pd.read_csv(a.table / "pairs.csv"))
    a.out.mkdir(parents=True, exist_ok=True)
    g.to_csv(a.out / "ddinter_pairs.csv", index=False)
    names_in = set(dd["a"]) | set(dd["b"])
    rep = {"ddinter_rows": int(len(dd)), "ddinter_distinct_drugs": len(names_in), "our_names_in_ddinter": len(set(nm) & names_in),
           "our_drugs_found": int(len(set(g.drug_a) | set(g.drug_b))), "pairs_among_our_drugs": int(len(g)),
           "of_which_in_twosides": int(g.in_twosides.sum()) if "in_twosides" in g else None,
           "of_which_NOT_in_twosides": int((~g.in_twosides).sum()) if "in_twosides" in g else None,
           "severity": g.severity.value_counts().to_dict(), "ambiguous_names_discarded": n_amb}
    (a.out / "ddinter_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
