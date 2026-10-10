"""Parse the openFDA bulk drug-label files (14 zip parts, ~1.9 GB) for the 645 drugs.

    python -m src.evidence.openfda_labels \
        --names /content/drive/MyDrive/auditddi-data/p1_targets/drug_names_drugcentral.csv \
        --pubchem-names /content/drive/MyDrive/auditddi-data/dataset_audit/drug_names_pubchem.csv \
        --labels /content/drive/MyDrive/auditddi-data/openfda_bulk_labels \
        --out /content/drive/MyDrive/auditddi-data/evidence_v1

Streaming: each part is read in 1 MB chunks and records are decoded one at a time, so memory stays small.

Matching (no guessing):
  * a label is attributed to one of our drugs only if one of its generic_name / substance_name entries
    EQUALS (case-insensitive) an unambiguous name of that drug, and the label is single-ingredient;
  * names that belong to two of our drugs are discarded;  names shorter than --min-name-len characters
    are not used to find MENTIONS (they match ordinary words).

Two products, with different scientific roles:
  label_features.csv  CYP substrate / inhibitor / inducer flags from the clinical-pharmacology text
                      -> drug-level FEATURES (annotation-rich scenario only: only marketed drugs have labels).
                      These are a keyword heuristic, not curated annotations; coverage is reported.
  label_pairs.csv     "the label of drug A names drug B in its drug-interactions text"
                      -> pair-level EVIDENCE for external validation.  NEVER a training feature.
"""
from __future__ import annotations

import argparse
import codecs
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

CYP_ISOFORMS = ["1A2", "2B6", "2C8", "2C9", "2C19", "2D6", "2E1", "3A4"]
_CYP_RE = re.compile(r"CYP\s*-?\s*(1A2|2B6|2C8|2C9|2C19|2D6|2E1|3A4)", re.I)
_SUB = re.compile(r"metaboli[sz]|substrate|oxidi[sz]|biotransform|catalyz|catalys", re.I)
_INH = re.compile(r"inhibit", re.I)
_IND = re.compile(r"induc", re.I)
_WS = re.compile(r"[ \n\r\t,]*")
_RESULTS = re.compile(r'"results"\s*:\s*\[')
KEEP = ("openfda", "drug_interactions", "clinical_pharmacology", "pharmacokinetics", "mechanism_of_action")


def iter_label_records(fobj, chunk: int = 1 << 20) -> Iterator[dict]:
    """Yield the objects of the top-level "results" array of an openFDA bulk file without loading the file."""
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    jd = json.JSONDecoder()
    buf, pos, started = "", 0, False
    while True:
        data = fobj.read(chunk)
        eof = not data
        buf += dec.decode(data, final=eof)
        if not started:
            m = _RESULTS.search(buf)
            if not m:
                if eof:
                    return
                continue
            pos, started = m.end(), True
        while True:
            pos = _WS.match(buf, pos).end()
            if pos >= len(buf):
                break
            if buf[pos] == "]":
                return
            try:
                obj, end = jd.raw_decode(buf, pos)
            except json.JSONDecodeError:
                break                                        # record continues in the next chunk
            yield obj
            pos = end
        buf, pos = buf[pos:], 0
        if eof:
            return


def iter_zip_records(zip_path: Path) -> Iterator[dict]:
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.lower().endswith(".json"):
                with z.open(name) as f:
                    yield from iter_label_records(f)


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def build_name_map(names: pd.DataFrame) -> Tuple[Dict[str, int], int]:
    """name -> drug_idx for names that belong to exactly one drug.  Returns (map, n_ambiguous_names)."""
    owners: Dict[str, set] = {}
    for r in names.itertuples():
        n = norm(r.name)
        if n and n != "nan":
            owners.setdefault(n, set()).add(int(r.drug_idx))
    unique = {k: next(iter(v)) for k, v in owners.items() if len(v) == 1}
    return unique, len(owners) - len(unique)


def _as_list(x) -> List[str]:
    if x is None:
        return []
    return [str(v) for v in x] if isinstance(x, list) else [str(x)]


def attribute_label(rec: dict, name_map: Dict[str, int]) -> Optional[int]:
    """drug_idx the label belongs to, or None.  Single-ingredient labels only."""
    ofda = rec.get("openfda") or {}
    subs = {norm(s) for s in _as_list(ofda.get("substance_name"))}
    gens = [norm(g) for g in _as_list(ofda.get("generic_name"))]
    if len(subs) > 1:
        return None                                          # combination product
    if not subs and any(("," in g or " and " in g or "/" in g) for g in gens):
        return None
    hits = {name_map[n] for n in list(subs) + gens if n in name_map}
    return next(iter(hits)) if len(hits) == 1 else None


def cyp_flags(text: str, window: int = 100) -> Dict[str, int]:
    """Keyword heuristic: for each CYP mention, look at +-window characters for substrate / inhibitor / inducer words."""
    flags = {f"cyp{i}_{role}": 0 for i in CYP_ISOFORMS for role in ("substrate", "inhibitor", "inducer")}
    for m in _CYP_RE.finditer(text):
        iso = m.group(1).upper()
        ctx = text[max(0, m.start() - window): m.end() + window]
        if _SUB.search(ctx):
            flags[f"cyp{iso}_substrate"] = 1
        if _INH.search(ctx):
            flags[f"cyp{iso}_inhibitor"] = 1
        if _IND.search(ctx):
            flags[f"cyp{iso}_inducer"] = 1
    return flags


def mention_regex(name_map: Dict[str, int], min_len: int) -> Optional[re.Pattern]:
    names = sorted((n for n in name_map if len(n) >= min_len), key=len, reverse=True)
    if not names:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I)


def find_mentions(text: str, rx: re.Pattern, name_map: Dict[str, int], own: int, snippet: int = 90) -> Dict[int, Tuple[int, str]]:
    out: Dict[int, Tuple[int, str]] = {}
    for m in rx.finditer(text):
        j = name_map.get(norm(m.group(1)))
        if j is None or j == own:
            continue
        cnt, snip = out.get(j, (0, text[max(0, m.start() - snippet): m.end() + snippet].replace("\n", " ")))
        out[j] = (cnt + 1, snip)
    return out


def process(records: Iterable[dict], name_map: Dict[str, int], min_name_len: int = 5, max_chars: int = 400_000):
    rx = mention_regex(name_map, min_name_len)
    labels: Dict[int, dict] = {}
    pairs: Dict[Tuple[int, int], Tuple[int, str]] = {}
    n_records = n_attr = 0
    for rec in records:
        n_records += 1
        d = attribute_label(rec, name_map)
        if d is None:
            continue
        n_attr += 1
        L = labels.setdefault(d, {"n_labels": 0, "pk": "", "di": ""})
        L["n_labels"] += 1
        if len(L["pk"]) < max_chars:
            L["pk"] += " " + " ".join(_as_list(rec.get("clinical_pharmacology")) + _as_list(rec.get("pharmacokinetics")))
        if len(L["di"]) < max_chars:
            L["di"] += " " + " ".join(_as_list(rec.get("drug_interactions")))
    feats, rows = [], []
    for d, L in sorted(labels.items()):
        feats.append({"drug_idx": d, "n_labels": L["n_labels"], "pk_text_chars": len(L["pk"]), **cyp_flags(L["pk"])})
        if rx is not None:
            for j, (c, snip) in find_mentions(L["di"], rx, name_map, d).items():
                rows.append({"drug_label": d, "drug_mentioned": j, "n_mentions": c, "snippet": snip})
    return pd.DataFrame(feats), pd.DataFrame(rows, columns=["drug_label", "drug_mentioned", "n_mentions", "snippet"]), {"records_seen": n_records, "records_attributed": n_attr}


def names_table(dc: pd.DataFrame, pubchem: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    parts = [dc[["drug_idx", "inn"]].rename(columns={"inn": "name"})] if "inn" in dc else []
    if pubchem is not None:
        if "title" in pubchem:
            parts.append(pubchem[["drug_idx", "title"]].rename(columns={"title": "name"}))
    return pd.concat(parts, ignore_index=True).dropna(subset=["name"]).drop_duplicates()


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True, type=Path)
    ap.add_argument("--pubchem-names", type=Path, default=None)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-name-len", type=int, default=5)
    ap.add_argument("--limit-parts", type=int, default=None, help="process only the first N zip parts (for a quick test)")
    a = ap.parse_args(list(argv) if argv is not None else None)

    pc = pd.read_csv(a.pubchem_names) if a.pubchem_names and a.pubchem_names.exists() else None
    nm, n_amb = build_name_map(names_table(pd.read_csv(a.names), pc))
    parts = sorted(a.labels.glob("*.zip"))[: a.limit_parts]
    feats, pairs, stats = [], [], {"records_seen": 0, "records_attributed": 0}
    # records of all parts are processed part by part and merged, so memory is bounded by one part
    for p in parts:
        f, pr, st = process(iter_zip_records(p), nm, a.min_name_len)
        feats.append(f); pairs.append(pr)
        for k in stats:
            stats[k] += st[k]
        print(f"{p.name}: {st['records_seen']} records, {st['records_attributed']} attributed to our drugs")
    a.out.mkdir(parents=True, exist_ok=True)
    F = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    if len(F):
        num = [c for c in F.columns if c not in ("drug_idx",)]
        F = F.groupby("drug_idx", as_index=False)[num].agg({c: ("sum" if c in ("n_labels", "pk_text_chars") else "max") for c in num})
    P = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=["drug_label", "drug_mentioned", "n_mentions", "snippet"])
    if len(P):
        P = P.groupby(["drug_label", "drug_mentioned"], as_index=False).agg(n_mentions=("n_mentions", "sum"), snippet=("snippet", "first"))
    F.to_csv(a.out / "label_features.csv", index=False)
    P.to_csv(a.out / "label_pairs.csv", index=False)
    cyp_cols = [c for c in F.columns if c.startswith("cyp")]
    rep = {**stats, "parts": len(parts), "names_ambiguous_discarded": n_amb, "drugs_with_a_label": int(len(F)),
           "drugs_with_any_cyp_flag": int((F[cyp_cols].sum(axis=1) > 0).sum()) if len(F) and cyp_cols else 0,
           "directed_mention_pairs": int(len(P)),
           "unordered_mention_pairs": int(len({tuple(sorted(x)) for x in zip(P.drug_label, P.drug_mentioned)})) if len(P) else 0}
    (a.out / "label_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
