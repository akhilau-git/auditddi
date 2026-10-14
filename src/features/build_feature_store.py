"""Build the honest drug feature store from real annotations.

    python -m src.features.build_feature_store \
        --table   /content/drive/MyDrive/auditddi-data/event_table_v1 \
        --ecfp    /content/drive/MyDrive/auditddi-data/baselines_v1/ecfp6_r3_1024.npz \
        --targets /content/drive/MyDrive/auditddi-data/p1_targets/drug_targets_chembl_mechanism.csv \
        --pharmgkb /content/drive/MyDrive/auditddi-data/pharmgkb_twosides_gene_profiles.csv \
        --out     /content/drive/MyDrive/auditddi-data/features_v1/drug_features.npz

Prints and saves the coverage report (next to --out as .json) with input SHA-256s.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.data_prep.build_event_table import load_event_table
from src.features.drug_features import build_store, rdkit_canonical, save_store, smiles_hash


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--ecfp", required=True, type=Path)
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--pharmgkb", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--atc", type=Path, default=None, help="drug_atc.csv (drug_idx, atc_code) from src.features.rxnorm_by_name")
    ap.add_argument("--cyp", type=Path, default=None, help="label_features.csv from src.evidence.openfda_labels")
    ap.add_argument("--esm", type=Path, default=None, help="drug_esm.npz from src.features.esm_embeddings")
    ap.add_argument("--human-only-targets", action="store_true")
    ap.add_argument("--identity-canon", action="store_true", help="TESTING ONLY: skip RDKit canonicalisation")
    a = ap.parse_args(list(argv) if argv is not None else None)

    table_p = a.table
    ecfp_p = a.ecfp
    if ecfp_p.is_dir():
        for cand in [ecfp_p / "ecfp6_r3_1024.npz", ecfp_p / "ecfp.npz"]:
            if cand.is_file():
                ecfp_p = cand
                break

    targets_p = a.targets
    if targets_p.is_dir():
        for cand in [
            targets_p / "drug_targets_chembl_mechanism.csv",
            targets_p / "drug_targets_union.csv",
            targets_p / "drug_targets_drugcentral.csv",
            targets_p / "targets.csv",
        ]:
            if cand.is_file():
                targets_p = cand
                break
        else:
            files = sorted(targets_p.rglob("drug_targets_*.csv")) or sorted(targets_p.rglob("*target*.csv"))
            if files:
                targets_p = files[0]

    pharmgkb_p = a.pharmgkb
    if pharmgkb_p.is_dir():
        for cand in [
            pharmgkb_p / "pharmgkb_twosides_gene_profiles.csv",
            pharmgkb_p / "expanded_profiles.csv",
            pharmgkb_p / "gene_profiles.csv",
        ]:
            if cand.is_file():
                pharmgkb_p = cand
                break
        else:
            files = sorted(pharmgkb_p.rglob("*profile*.csv")) or sorted(pharmgkb_p.rglob("*gene*.csv"))
            if files:
                pharmgkb_p = files[0]

    cyp_p = a.cyp
    if cyp_p is not None and cyp_p.is_dir():
        for cand in [
            cyp_p / "label_features.csv",
            cyp_p / "cyp_features.csv",
        ]:
            if cand.is_file():
                cyp_p = cand
                break
        else:
            files = sorted(cyp_p.rglob("label_features.csv")) or sorted(cyp_p.rglob("*cyp*.csv"))
            if files:
                cyp_p = files[0]

    atc_p = a.atc
    if atc_p is not None and atc_p.is_dir():
        cand = atc_p / "drug_atc.csv"
        if cand.is_file():
            atc_p = cand

    out_p = a.out
    if out_p.is_dir() or not out_p.suffix:
        out_p = out_p / "drug_features.npz"

    table = load_event_table(table_p)
    z = np.load(ecfp_p, allow_pickle=False)
    if str(z["smiles_sha256"]) != smiles_hash(table.drugs["smiles"].tolist()):
        raise ValueError("ECFP cache was built for a different drug list")
    targets = pd.read_csv(targets_p)
    profiles = pd.read_csv(pharmgkb_p)
    canon = (lambda s: s) if a.identity_canon else rdkit_canonical
    atc = pd.read_csv(atc_p) if atc_p and atc_p.is_file() else None
    cyp = pd.read_csv(cyp_p) if cyp_p and cyp_p.is_file() else None
    esm = dict(np.load(a.esm, allow_pickle=False)) if a.esm else None
    store = build_store(table.drugs, z["bits"], z["valid"], targets, profiles, canon_fn=canon, human_only_targets=a.human_only_targets, atc=atc, cyp=cyp, esm=esm)
    rep = save_store(store, out_p, extra={"targets": {"path": str(targets_p), "sha256": _sha(targets_p)},
                                          "pharmgkb": {"path": str(pharmgkb_p), "sha256": _sha(pharmgkb_p)},
                                          "ecfp": {"path": str(ecfp_p), "sha256": _sha(ecfp_p)}})
    for k, v in rep.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
