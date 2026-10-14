"""Sequences for every target accession in the union target table (ChEMBL + DrugCentral).

    python -m src.features.uniprot_sequences \
        --targets /content/drive/MyDrive/auditddi-data/p1_targets/drug_targets_union.csv \
        --local-fasta /content/drive/MyDrive/auditddi-data/uniprot \
        --out /content/drive/MyDrive/auditddi-data/uniprot_v2

Order of preference: local FASTA files first, then the UniProt REST API for what is missing.
An accession UniProt cannot return stays missing and is listed; nothing is substituted.
Outputs: target_sequences.csv (accession, length, organism, sha256, source) and target_sequences.fasta.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

FASTA_HDR = re.compile(r"^>(?:sp|tr)\|([A-Z0-9]+(?:-\d+)?)\|(\S+)(?:\s+(.*))?$")


def parse_fasta(text: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    acc, hdr, seq = None, "", []

    def flush():
        if acc is not None:
            org = re.search(r"OS=(.*?)(?:\sOX=|\sGN=|\sPE=|$)", hdr)
            s = "".join(seq).upper()
            out[acc] = {"header": hdr, "sequence": s, "length": len(s), "organism": org.group(1) if org else ""}

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


def load_local(fasta_dir: Path) -> Dict[str, dict]:
    found: Dict[str, dict] = {}
    for p in sorted(Path(fasta_dir).rglob("*.fasta")):
        try:
            for k, v in parse_fasta(p.read_text()).items():
                found.setdefault(k, {**v, "source": "local"})
        except Exception:
            continue
    return found


def fetch(accessions: List[str], session, batch: int = 100, pause: float = 0.3) -> Dict[str, dict]:
    got: Dict[str, dict] = {}
    for i in range(0, len(accessions), batch):
        chunk = accessions[i:i + batch]
        txt = ""
        try:
            r = session.get("https://rest.uniprot.org/uniprotkb/accessions", params={"accessions": ",".join(chunk), "format": "fasta"}, timeout=60)
            if r.status_code == 200 and r.text.startswith(">"):
                txt = r.text
        except Exception:
            pass
        if not txt:                                           # fall back to one request per accession
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
        for k, v in parse_fasta(txt).items():
            got[k] = {**v, "source": "uniprot_rest"}
        time.sleep(pause)
    return got


def collect(accessions: Sequence[str], local: Dict[str, dict], session=None) -> Dict[str, dict]:
    have = {a: local[a] for a in accessions if a in local}
    missing = [a for a in accessions if a not in have]
    if missing and session is not None:
        have.update(fetch(missing, session))
    return have


def write_outputs(seqs: Dict[str, dict], wanted: Sequence[str], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"accession": a, "length": v["length"], "organism": v["organism"], "source": v["source"],
             "sha256": hashlib.sha256(v["sequence"].encode()).hexdigest()} for a, v in sorted(seqs.items())]
    pd.DataFrame(rows, columns=["accession", "length", "organism", "source", "sha256"]).to_csv(out / "target_sequences.csv", index=False)
    with open(out / "target_sequences.fasta", "w") as f:
        for a, v in sorted(seqs.items()):
            f.write(f">{a}\n" + "\n".join(v["sequence"][j:j + 60] for j in range(0, len(v["sequence"]), 60)) + "\n")
    missing = sorted(set(wanted) - set(seqs))
    rep = {"accessions_needed": len(set(wanted)), "with_sequence": len(seqs), "missing": len(missing), "missing_examples": missing[:25],
           "from_local": sum(v["source"] == "local" for v in seqs.values()), "from_rest": sum(v["source"] == "uniprot_rest" for v in seqs.values()),
           "coverage": round(len(seqs) / max(len(set(wanted)), 1), 4)}
    (out / "uniprot_report.json").write_text(json.dumps(rep, indent=2))
    return rep


def main(argv: Optional[Sequence[str]] = None, session=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--local-fasta", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--offline", action="store_true", help="do not call the UniProt API")
    a = ap.parse_args(list(argv) if argv is not None else None)
    wanted = sorted(pd.read_csv(a.targets).uniprot.dropna().astype(str).unique())
    local = load_local(a.local_fasta) if a.local_fasta else {}
    if session is None and not a.offline:
        import requests
        session = requests.Session()
        session.headers["User-Agent"] = "AuditDDI-uniprot/1.0"
    rep = write_outputs(collect(wanted, local, None if a.offline else session), wanted, a.out)
    for k, v in rep.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
