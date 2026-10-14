"""AuditDDI dataset audit (Guide 1, section 3 / milestone M2).

Self-contained: pandas, numpy, requests (+ RDKit for InChIKeys).  READ-ONLY on your
existing data; everything it writes goes to  <ROOT>/dataset_audit/ .

Sections (each returns plain dicts so a failure in one never stops the others):
  local_checks      what is on Drive for every Guide 1 source, and its real coverage
  uniprot           which ChEMBL target accessions have a sequence; fetches the missing ones
  pubchem_names     names for your 645 drugs (the bridge DDInter and RxNorm need)
  ddinter_bridge    how many DDInter pairs involve your drugs, and overlap with TWOSIDES pairs
  api_checks        is each online source reachable from this runtime
  summary           one table: source / role / status / coverage / action
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/content/drive/MyDrive/auditddi-data")
EXPECTED_DDINTER_LETTERS = list("ABCDGHJLMNPRSV")    # ATC level-1 groups (DDInter downloads are split this way)


def out_dir(root: Path) -> Path:
    d = Path(root) / "dataset_audit"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, max_mb: float = 200.0) -> Optional[str]:
    path = Path(path)
    if path.stat().st_size > max_mb * 1e6:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inventory(root: Path, subdirs: Iterable[str]) -> pd.DataFrame:
    rows = []
    for s in subdirs:
        base = Path(root) / s
        if not base.exists():
            rows.append({"source_dir": s, "path": "(folder missing)", "size_mb": 0.0, "sha256": None})
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rows.append({"source_dir": s, "path": str(p.relative_to(root)), "size_mb": round(p.stat().st_size / 1e6, 3), "sha256": sha256_file(p)})
    return pd.DataFrame(rows)


def rdkit_inchikey(smiles: str) -> Optional[str]:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import inchi  # type: ignore

    m = Chem.MolFromSmiles(str(smiles))
    return inchi.MolToInchiKey(m) if m is not None else None


def rdkit_canonical(smiles: str) -> Optional[str]:
    from rdkit import Chem  # type: ignore

    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m is not None else None


# --------------------------------------------------------------------------- #
# UniProt
# --------------------------------------------------------------------------- #
FASTA_HDR = re.compile(r"^>(?:sp|tr)\|([A-Z0-9]+(?:-\d+)?)\|(\S+)(?:\s+(.*))?$")


def parse_fasta(text: str) -> Dict[str, dict]:
    """accession -> {entry, length, organism, header}.  Works for UniProt headers."""
    out: Dict[str, dict] = {}
    acc, hdr, seq = None, None, []

    def flush():
        if acc is not None:
            m = FASTA_HDR.match(hdr)
            org = re.search(r"OS=(.*?)(?:\sOX=|\sGN=|\sPE=|$)", hdr)
            out[acc] = {"entry": m.group(2) if m else "", "length": len("".join(seq)), "organism": org.group(1) if org else "",
                        "sequence": "".join(seq), "header": hdr}

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            hdr, seq = line.strip(), []
            m = FASTA_HDR.match(hdr)
            acc = m.group(1) if m else hdr[1:].split()[0]
        elif acc is not None:
            seq.append(line.strip())
    flush()
    return out


def scan_uniprot_dir(d: Path) -> Dict[str, dict]:
    found: Dict[str, dict] = {}
    for p in sorted(Path(d).rglob("*.fasta")):
        try:
            for k, v in parse_fasta(p.read_text()).items():
                v.pop("sequence", None)
                v["file"] = p.name
                found.setdefault(k, v)
        except Exception:
            continue
    return found


def fetch_uniprot_fasta(accessions: List[str], session, batch: int = 100, pause: float = 0.3) -> Dict[str, dict]:
    """Batch download of FASTA from the UniProt REST API; falls back per accession."""
    got: Dict[str, dict] = {}
    for i in range(0, len(accessions), batch):
        chunk = accessions[i:i + batch]
        txt = None
        try:
            r = session.get("https://rest.uniprot.org/uniprotkb/accessions", params={"accessions": ",".join(chunk), "format": "fasta"}, timeout=60)
            if r.status_code == 200 and r.text.startswith(">"):
                txt = r.text
        except Exception:
            pass
        if txt is None:
            parts = []
            for a in chunk:
                try:
                    r = session.get(f"https://rest.uniprot.org/uniprotkb/{a}.fasta", timeout=30)
                    if r.status_code == 200 and r.text.startswith(">"):
                        parts.append(r.text)
                except Exception:
                    pass
                time.sleep(pause / 3)
            txt = "\n".join(parts)
        got.update(parse_fasta(txt))
        time.sleep(pause)
    return got


def uniprot_section(root: Path, session=None, fetch: bool = True) -> dict:
    root = Path(root)
    tg_path = root / "p1_targets" / "drug_targets_chembl_mechanism.csv"
    res: dict = {"targets_csv": tg_path.exists()}
    if not tg_path.exists():
        res["note"] = "run the ChEMBL target step first (p1_targets/drug_targets_chembl_mechanism.csv)"
        return res
    need = sorted(pd.read_csv(tg_path).uniprot.dropna().astype(str).unique())
    have_local = scan_uniprot_dir(root / "uniprot")
    in_local = [a for a in need if a in have_local]
    missing = [a for a in need if a not in have_local]
    res.update({"target_accessions_needed": len(need), "already_in_local_fasta": len(in_local), "missing": len(missing),
                "local_fasta_accessions": len(have_local)})
    fetched: Dict[str, dict] = {}
    if fetch and missing and session is not None:
        fetched = fetch_uniprot_fasta(missing, session)
        d = out_dir(root)
        with open(d / "chembl_targets_fetched.fasta", "w") as f:
            for a, v in fetched.items():
                f.write(f"{v['header']}\n" + "\n".join(v["sequence"][j:j + 60] for j in range(0, len(v["sequence"]), 60)) + "\n")
        pd.DataFrame([{"accession": a, "entry": v["entry"], "length": v["length"], "organism": v["organism"]} for a, v in fetched.items()]).to_csv(d / "chembl_targets_fetched.csv", index=False)
    res["fetched_now"] = len(fetched)
    res["still_missing"] = sorted(set(missing) - set(fetched))[:50]
    res["n_still_missing"] = len(set(missing) - set(fetched))
    res["sequence_coverage_of_targets"] = round((len(in_local) + len(fetched)) / max(len(need), 1), 4)
    return res


# --------------------------------------------------------------------------- #
# PubChem names
# --------------------------------------------------------------------------- #
def _get_json(session, url, tries: int = 4, pause: float = 0.3):
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


def pubchem_names(root: Path, session, inchikey_fn: Callable[[str], Optional[str]] = rdkit_inchikey, pause: float = 0.25) -> pd.DataFrame:
    """One row per drug: pubchem CID, Title, up to 30 short synonyms.  Cached on Drive."""
    root = Path(root)
    drugs = pd.read_csv(root / "event_table_v1" / "drugs.csv")
    cache_f = out_dir(root) / "pubchem_names_cache.json"
    cache = json.load(open(cache_f)) if cache_f.exists() else {}
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
            json.dump(cache, open(cache_f, "w"))
        time.sleep(pause)
    json.dump(cache, open(cache_f, "w"))
    rows = [{"drug_idx": int(k), **v, "synonyms": "|".join(v["synonyms"])} for k, v in cache.items()]
    df = pd.DataFrame(rows).sort_values("drug_idx")
    df.to_csv(out_dir(root) / "drug_names_pubchem.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# DDInter
# --------------------------------------------------------------------------- #
def ddinter_inventory(root: Path) -> dict:
    raw = Path(root) / "ddinter" / "raw"
    files = sorted(raw.glob("ddinter_downloads_code_*.csv")) if raw.exists() else []
    letters = sorted(re.search(r"code_([A-Z])", f.name).group(1) for f in files)
    total = 0
    for f in files:
        try:
            total += sum(1 for _ in open(f, "rb")) - 1
        except Exception:
            pass
    return {"files": len(files), "letters_present": letters,
            "letters_missing": [c for c in EXPECTED_DDINTER_LETTERS if c not in letters], "rows": int(total)}


def load_ddinter_raw(root: Path) -> pd.DataFrame:
    parts = [pd.read_csv(f) for f in sorted((Path(root) / "ddinter" / "raw").glob("ddinter_downloads_code_*.csv"))]
    df = pd.concat(parts, ignore_index=True)
    df["a"] = df["Drug_A"].astype(str).str.lower().str.strip()
    df["b"] = df["Drug_B"].astype(str).str.lower().str.strip()
    return df


def ddinter_overlap(ddinter: pd.DataFrame, names: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """names: drug_idx, title, synonyms('|'-joined).  pairs: drug_a, drug_b (TWOSIDES pairs)."""
    name_to_drugs: Dict[str, set] = {}
    for r in names.itertuples():
        cands = [str(r.title)] if isinstance(r.title, str) else []
        cands += [x for x in str(r.synonyms).split("|") if x and x != "nan"]
        for c in cands:
            name_to_drugs.setdefault(c.lower().strip(), set()).add(int(r.drug_idx))
    # only unambiguous names (one drug) are used, so a shared synonym cannot create a wrong match
    unique = {k: next(iter(v)) for k, v in name_to_drugs.items() if len(v) == 1}
    da = ddinter["a"].map(unique)
    db = ddinter["b"].map(unique)
    both = ddinter[da.notna() & db.notna()].copy()
    both["da"], both["db"] = da[both.index].astype(int), db[both.index].astype(int)
    both = both[both.da != both.db]
    both["lo"], both["hi"] = both[["da", "db"]].min(axis=1), both[["da", "db"]].max(axis=1)
    uniq_pairs = both.drop_duplicates(["lo", "hi"])
    two = set(zip(np.minimum(pairs.drug_a, pairs.drug_b), np.maximum(pairs.drug_a, pairs.drug_b)))
    in_two = sum((int(a), int(b)) in two for a, b in zip(uniq_pairs.lo, uniq_pairs.hi))
    matched_drugs = set(da.dropna().astype(int)) | set(db.dropna().astype(int))
    return {"ddinter_rows": int(len(ddinter)), "our_drugs_found_in_ddinter": len(matched_drugs),
            "ddinter_pairs_with_both_drugs_ours": int(len(uniq_pairs)), "of_which_in_twosides_pairs": int(in_two),
            "severity_of_those_pairs": uniq_pairs["Level"].value_counts().to_dict() if "Level" in uniq_pairs else {}}


# --------------------------------------------------------------------------- #
# online reachability
# --------------------------------------------------------------------------- #
API_CHECKS = [
    ("UniProt", "https://rest.uniprot.org/uniprotkb/P05067.fasta", lambda r: r.text.startswith(">")),
    ("ChEMBL", "https://www.ebi.ac.uk/chembl/api/data/status.json", lambda r: "chembl_db_version" in r.text),
    ("PubChem", "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/cids/JSON", lambda r: "IdentifierList" in r.text),
    ("RxNorm", "https://rxnav.nlm.nih.gov/REST/rxcui.json?name=aspirin", lambda r: "idGroup" in r.text),
    ("openFDA", "https://api.fda.gov/drug/label.json?search=openfda.generic_name:aspirin&limit=1", lambda r: "results" in r.text),
    ("DailyMed", "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name=aspirin&pagesize=1", lambda r: "data" in r.text),
    ("KEGG", "https://rest.kegg.jp/find/drug/aspirin", lambda r: len(r.text.strip()) > 0),
    ("DrugCentral (site)", "https://drugcentral.org/", lambda r: r.status_code == 200),
]


def api_checks(session, checks=API_CHECKS) -> pd.DataFrame:
    rows = []
    for name, url, ok in checks:
        t0 = time.time()
        try:
            r = session.get(url, timeout=25)
            good = r.status_code == 200 and bool(ok(r))
            rows.append({"source": name, "reachable": good, "http": r.status_code, "seconds": round(time.time() - t0, 2),
                         "note": "" if good else "responded but unexpected content" if r.status_code == 200 else f"HTTP {r.status_code}"})
        except Exception as e:  # noqa: BLE001
            rows.append({"source": name, "reachable": False, "http": None, "seconds": round(time.time() - t0, 2), "note": type(e).__name__})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# local checks
# --------------------------------------------------------------------------- #
def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def local_checks(root: Path, canon_fn: Callable[[str], Optional[str]] = rdkit_canonical) -> dict:
    root = Path(root)
    res: dict = {}
    # TWOSIDES
    mf = root / "event_table_v1" / "manifest.json"
    if mf.exists():
        rep = json.loads(mf.read_text())["report"]
        res["TWOSIDES"] = {"present": True, **{k: rep[k] for k in ("n_drugs", "n_pairs", "n_events_all", "n_positive_pair_event_cells")}}
    else:
        res["TWOSIDES"] = {"present": False}
    drugs = pd.read_csv(root / "event_table_v1" / "drugs.csv") if (root / "event_table_v1" / "drugs.csv").exists() else None
    canon_set = {canon_fn(s) for s in drugs.smiles} if drugs is not None else set()
    # PharmGKB
    pg = root / "pharmgkb"
    res["PharmGKB"] = {"present": pg.exists(), "has_relationships": (pg / "relationships.tsv").exists(),
                       "has_chemicals": (pg / "chemicals.tsv").exists(), "has_genes": (pg / "genes.tsv").exists()}
    prof = root / "pharmgkb_twosides_gene_profiles.csv"
    if prof.exists() and drugs is not None:
        d = pd.read_csv(prof)
        res["PharmGKB"]["drugs_with_real_gene_profile"] = len({canon_fn(s) for s in d.canonical_smiles} & canon_set)
    # BindingDB
    bd = root / "BindingDB" / "drug_target_edges.csv"
    if bd.exists() and drugs is not None:
        e = pd.read_csv(bd)
        c = e.source.map(canon_fn)
        res["BindingDB"] = {"present": True, "rows": int(len(e)), "drugs_matched": int(c[c.isin(canon_set)].nunique()), "distinct_targets": int(e.target.nunique())}
    else:
        res["BindingDB"] = {"present": bd.exists()}
    # PDB, FAERS, GEO, PubChem, ChEMBL
    res["PDB"] = {"structure_files": len(list((root / "PDB").rglob("*.pdb"))) if (root / "PDB").exists() else 0}
    fa = sorted((root / "faers").rglob("DRUG*.txt")) if (root / "faers").exists() else []
    res["FAERS"] = {"quarters": sorted({re.sub(r"\D*(\d\dQ\d).*", r"\1", f.name) for f in fa})}
    ge = sorted(p.name for p in (root / "GEO").glob("*")) if (root / "GEO").exists() else []
    res["GEO"] = {"files": ge}
    res["PubChem"] = {"files": sorted(p.name for p in (root / "PubChem").glob("*")) if (root / "PubChem").exists() else []}
    tg = root / "p1_targets" / "drug_targets_chembl_mechanism.csv"
    res["ChEMBL"] = {"structure_file": bool(list((root / "chembl").glob("chembl_*_chemreps.txt*"))) if (root / "chembl").exists() else False,
                     "mechanism_targets_csv": tg.exists()}
    if tg.exists():
        t = pd.read_csv(tg)
        res["ChEMBL"].update({"drugs_with_real_targets": int(t.drug_idx.nunique()), "distinct_target_accessions": int(t.uniprot.nunique())})
    res["DDInter"] = ddinter_inventory(root)
    res["DrugCentral"] = {"present": any(root.glob("*rugcentral*")) or (root / "drugcentral").exists()}
    for s, folder in (("KEGG", "kegg"), ("RxNorm", "rxnorm"), ("openFDA", "openfda"), ("DailyMed", "dailymed")):
        res[s] = {"present": (root / folder).exists()}
    return res


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def summary(local: dict, uni: Optional[dict], ddi: Optional[dict], apis: Optional[pd.DataFrame], n_drugs: int = 645) -> pd.DataFrame:
    up = {r.source: r.reachable for r in apis.itertuples()} if apis is not None else {}
    rows = []

    def add(src, role, status, cover, action):
        rows.append({"source": src, "guide1_role": role, "status": status, "coverage_of_drugs": cover, "action": action})

    t = local.get("TWOSIDES", {})
    add("TWOSIDES", "primary labels", "OK" if t.get("n_events_all", 0) >= 1300 and t.get("n_drugs") == n_drugs else "CHECK",
        f"{t.get('n_drugs', '?')} drugs / {t.get('n_pairs', '?')} pairs / {t.get('n_events_all', '?')} events", "none")
    p = local["PharmGKB"]
    n = p.get("drugs_with_real_gene_profile", 0)
    add("PharmGKB", "enzyme/gene", "PARTIAL" if n else "MISSING", f"{n}/{n_drugs}", "none unless higher coverage is needed")
    c = local["ChEMBL"]
    k = c.get("drugs_with_real_targets", 0)
    add("ChEMBL", "targets", "OK" if k >= 0.4 * n_drugs else "PARTIAL" if k else "MISSING", f"{k}/{n_drugs}", "none")
    if uni and "sequence_coverage_of_targets" in uni:
        sc = uni["sequence_coverage_of_targets"]
        add("UniProt", "protein sequences", "OK" if sc >= 0.95 else "PARTIAL", f"{sc:.0%} of {uni['target_accessions_needed']} target accessions",
            "none" if sc >= 0.95 else f"{uni.get('n_still_missing', '?')} accessions have no sequence; inspect still_missing")
    else:
        add("UniProt", "protein sequences", "UNKNOWN", "-", "run the uniprot section")
    b = local["BindingDB"]
    bm = b.get("drugs_matched", 0)
    add("BindingDB", "affinities", "OK" if bm >= 0.4 * n_drugs else "PARTIAL" if bm else "MISSING", f"{bm}/{n_drugs}", "full BindingDB download (optional)" if bm < 0.4 * n_drugs else "none")
    npdb = local["PDB"]["structure_files"]
    add("PDB", "structure (optional)", "MISSING" if npdb < 50 else "PARTIAL", f"{npdb} structure files", "optional; skip or document as future work")
    d = local["DDInter"]
    status = "OK" if not d["letters_missing"] and d["files"] else "PARTIAL" if d["files"] else "MISSING"
    cover = f"{d['files']}/{len(EXPECTED_DDINTER_LETTERS)} category files, {d['rows']} rows"
    if ddi:
        cover += f"; {ddi['ddinter_pairs_with_both_drugs_ours']} pairs among our drugs"
    add("DDInter", "curated evidence", status, cover, f"download categories {d['letters_missing']}" if d["letters_missing"] else "none")
    for s, role in (("DrugCentral", "targets/pharmacology"), ("KEGG", "curated evidence"), ("RxNorm", "identity"), ("openFDA", "label evidence"), ("DailyMed", "label evidence")):
        present = local[s]["present"]
        key = "DrugCentral (site)" if s == "DrugCentral" else s
        reach = up.get(key)
        add(s, role, "OK" if present else "MISSING", "-", "ready" if present else ("reachable online, fetch it" if reach else "download/connect needed" if reach is not None else "check connection"))
    add("PubChem", "identity", "PARTIAL", "-", "names fetched via API in section 3")
    q = local["FAERS"]["quarters"]
    add("FAERS", "optional auxiliary toxicity", "WEAK", f"quarters: {q}", "single quarter only; keep as optional ablation")
    add("GEO", "(not in Guide 1)", "NOT USED", "-", "disease-tissue series, not drug perturbation; drop from the model")
    return pd.DataFrame(rows)
