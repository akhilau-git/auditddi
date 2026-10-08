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
    ap.add_argument("--human-only-targets", action="store_true")
    ap.add_argument("--identity-canon", action="store_true", help="TESTING ONLY: skip RDKit canonicalisation")
    a = ap.parse_args(list(argv) if argv is not None else None)

    table = load_event_table(a.table)
    z = np.load(a.ecfp, allow_pickle=False)
    if str(z["smiles_sha256"]) != smiles_hash(table.drugs["smiles"].tolist()):
        raise ValueError("ECFP cache was built for a different drug list")
    targets = pd.read_csv(a.targets)
    profiles = pd.read_csv(a.pharmgkb)
    canon = (lambda s: s) if a.identity_canon else rdkit_canonical
    store = build_store(table.drugs, z["bits"], z["valid"], targets, profiles, canon_fn=canon, human_only_targets=a.human_only_targets)
    rep = save_store(store, a.out, extra={"targets": {"path": str(a.targets), "sha256": _sha(a.targets)},
                                          "pharmgkb": {"path": str(a.pharmgkb), "sha256": _sha(a.pharmgkb)},
                                          "ecfp": {"path": str(a.ecfp), "sha256": _sha(a.ecfp)}})
    for k, v in rep.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
