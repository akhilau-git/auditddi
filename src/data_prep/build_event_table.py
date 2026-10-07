"""Multi-label TWOSIDES event table (Guide 1, sections 4.7, 4.8, 5.1, 5.3).

Turns the event-level edge list (source SMILES, target SMILES, event name) into:

    pairs.csv        one row per UNORDERED drug pair   (pair_id, drug_a, drug_b)
    drugs.csv        one row per drug                  (drug_idx, smiles)
    events_all.csv   every observed event              (event_idx, event, n_pairs)
    labels.npz       sparse CSR matrix Y[n_pairs, n_events_all], 1 = event reported
    manifest.json    sha256 of every output + input, counts, parameters

Design rules taken from Guide 1:
  * A pair's complete event set always moves together (one row per pair).
  * The event vocabulary is frozen from the TRAINING partition only
    (``freeze_vocabulary``), never from test labels.
  * pos_weight_e = (#negatives_e / #positives_e) is computed on the training
    partition only (``pos_weights``).
  * A missing (pair, event) record is "unreported", not "safe".  The table
    stores only positives; negatives are implicit per event head.

Pure numpy / pandas / scipy.  RDKit canonicalisation is optional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

SCHEMA_VERSION = "event_table_v1"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalise_event(name: str) -> str:
    """Lower-case, collapse whitespace.  Applied identically to every row."""
    return " ".join(str(name).strip().lower().split())


def canonical_smiles(smiles: str) -> str:
    """RDKit canonical SMILES if RDKit is installed, otherwise the stripped
    input string.  Guide 1 section 10.5: never silently change identity, so the
    caller reports how many strings changed (see ``build_event_table``)."""
    s = str(smiles).strip()
    try:
        from rdkit import Chem  # type: ignore

        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else s
    except Exception:
        return s


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass
class EventTable:
    drugs: pd.DataFrame          # drug_idx, smiles
    pairs: pd.DataFrame          # pair_id, drug_a, drug_b  (drug_a < drug_b, indices)
    events: pd.DataFrame         # event_idx, event, n_pairs   (all observed events)
    Y: sp.csr_matrix             # [n_pairs, n_events_all] uint8

    @property
    def n_pairs(self) -> int:
        return self.Y.shape[0]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def _factorize_clean(col: pd.Series, fn) -> tuple[np.ndarray, list]:
    """Factorize ``col`` (object or category) and apply ``fn`` to the UNIQUE
    values only.  Returns (codes, cleaned_uniques); code -1 = missing.  Several
    raw values may map to the same cleaned value; codes are remapped so they
    do.  Works on 4.6M rows without materialising 4.6M Python strings."""
    codes, uniques = pd.factorize(col, sort=False)
    cleaned = [fn(u) for u in uniques]
    uniq_clean = sorted(set(cleaned))
    remap = {c: i for i, c in enumerate(uniq_clean)}
    lut = np.array([remap[c] for c in cleaned] + [-1], dtype=np.int64)  # last slot handles code -1
    out = lut[np.where(codes < 0, len(cleaned), codes)]
    return out, uniq_clean


def build_event_table(
    edges: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    event_col: str = "interaction_type",
    canonicalise: bool = False,
) -> tuple[EventTable, dict]:
    """Build the sparse pair x event matrix from an edge list.

    Duplicate (pair, event) rows and A-B / B-A repeats collapse to one positive.
    Self-pairs and rows with missing values are dropped and counted.
    """
    report: dict = {"rows_in": int(len(edges))}
    smiles_fn = (lambda x: canonical_smiles(x)) if canonicalise else (lambda x: str(x).strip())

    cs, us = _factorize_clean(edges[source_col], smiles_fn)
    ct, ut = _factorize_clean(edges[target_col], smiles_fn)
    ce, ue = _factorize_clean(edges[event_col], normalise_event)

    if canonicalise:
        raw = set(pd.unique(edges[source_col].dropna())) | set(pd.unique(edges[target_col].dropna()))
        report["distinct_smiles_before_canonicalisation"] = int(len(raw))
        report["distinct_smiles_after_canonicalisation"] = int(len(set(us) | set(ut)))

    ok = (cs >= 0) & (ct >= 0) & (ce >= 0)
    report["rows_dropped_missing"] = int((~ok).sum())
    cs, ct, ce = cs[ok], ct[ok], ce[ok]

    # shared, order-independent drug index
    drug_smiles = sorted(set(us) | set(ut))
    didx = {s_: i for i, s_ in enumerate(drug_smiles)}
    a = np.array([didx[x] for x in us], dtype=np.int64)[cs]
    b = np.array([didx[x] for x in ut], dtype=np.int64)[ct]

    nonself = a != b
    report["rows_dropped_self_pairs"] = int((~nonself).sum())
    a, b, ce = a[nonself], b[nonself], ce[nonself]

    lo, hi = np.minimum(a, b), np.maximum(a, b)
    n_drugs = len(drug_smiles)

    ev_used = np.unique(ce)                     # events that survive cleaning
    ev_names = [ue[i] for i in ev_used]         # already alphabetical (uniq_clean sorted)
    e = np.searchsorted(ev_used, ce)

    key = lo * n_drugs + hi
    uniq_key, pair_row = np.unique(key, return_inverse=True)
    n_pairs = len(uniq_key)

    Y = sp.coo_matrix(
        (np.ones(len(pair_row), dtype=np.int32), (pair_row, e)),
        shape=(n_pairs, len(ev_names)),
    ).tocsr()                                    # duplicates are summed here
    report["duplicate_rows_collapsed"] = int((Y.data - 1).clip(min=0).sum())
    Y.data = np.minimum(Y.data, 1)
    Y = Y.astype(np.uint8)
    Y.eliminate_zeros()

    pairs = pd.DataFrame(
        {
            "pair_id": np.arange(n_pairs, dtype=np.int64),
            "drug_a": (uniq_key // n_drugs).astype(np.int64),
            "drug_b": (uniq_key % n_drugs).astype(np.int64),
        }
    )
    drugs = pd.DataFrame({"drug_idx": np.arange(n_drugs), "smiles": drug_smiles})
    per_event = np.asarray(Y.sum(axis=0)).ravel().astype(np.int64)
    events = pd.DataFrame({"event_idx": np.arange(len(ev_names)), "event": ev_names, "n_pairs": per_event})

    ev_per_pair = np.asarray(Y.sum(axis=1)).ravel()
    report.update(
        {
            "n_drugs": int(n_drugs),
            "n_pairs": int(n_pairs),
            "n_events_all": int(len(ev_names)),
            "n_positive_pair_event_cells": int(Y.nnz),
            "min_events_per_pair": int(ev_per_pair.min()) if n_pairs else 0,
            "median_events_per_pair": float(np.median(ev_per_pair)) if n_pairs else 0.0,
        }
    )
    return EventTable(drugs=drugs, pairs=pairs, events=events, Y=Y), report


# --------------------------------------------------------------------------- #
# vocabulary + loss weights  (training partition ONLY)
# --------------------------------------------------------------------------- #
def freeze_vocabulary(
    Y: sp.csr_matrix,
    train_rows: Sequence[int],
    events: pd.DataFrame,
    min_train_pairs: int,
    min_train_negatives: int = 1,
) -> pd.DataFrame:
    """Return the frozen vocabulary computed from ``train_rows`` only.

    An event is kept if it has at least ``min_train_pairs`` positive training
    pairs and at least ``min_train_negatives`` negative training pairs.
    Event order in the returned frame = column order of the model head.
    """
    tr = np.asarray(train_rows, dtype=np.int64)
    pos = np.asarray(Y[tr].sum(axis=0)).ravel().astype(np.int64)
    neg = len(tr) - pos
    keep = (pos >= min_train_pairs) & (neg >= min_train_negatives)
    vocab = events.loc[keep, ["event_idx", "event"]].copy()
    vocab["train_pos"] = pos[keep]
    vocab["train_neg"] = neg[keep]
    vocab = vocab.reset_index(drop=True)
    vocab.insert(0, "head_idx", np.arange(len(vocab)))
    return vocab


def vocabulary_hash(vocab: pd.DataFrame) -> str:
    payload = "\n".join(f"{r.head_idx}\t{r.event_idx}\t{r.event}" for r in vocab.itertuples())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pos_weights(vocab: pd.DataFrame, clip: Optional[float] = None) -> np.ndarray:
    """pos_weight_e = negatives_e / positives_e (training partition only)."""
    w = vocab["train_neg"].to_numpy(dtype=np.float64) / np.maximum(vocab["train_pos"].to_numpy(dtype=np.float64), 1.0)
    if clip is not None:
        w = np.minimum(w, clip)
    return w.astype(np.float32)


def labels_for(Y: sp.csr_matrix, rows: Sequence[int], vocab: pd.DataFrame) -> np.ndarray:
    """Dense [len(rows), n_heads] float32 label block for the frozen vocabulary."""
    cols = vocab["event_idx"].to_numpy(dtype=np.int64)
    return Y[np.asarray(rows, dtype=np.int64)][:, cols].toarray().astype(np.float32)


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_event_table(table: EventTable, out_dir: Path, report: dict, input_path: Optional[Path] = None, params: Optional[dict] = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.drugs.to_csv(out_dir / "drugs.csv", index=False)
    table.pairs.to_csv(out_dir / "pairs.csv", index=False)
    table.events.to_csv(out_dir / "events_all.csv", index=False)
    sp.save_npz(out_dir / "labels.npz", table.Y)
    files = {p.name: sha256_file(p) for p in sorted(out_dir.glob("*")) if p.name != "manifest.json"}
    manifest = {
        "schema": SCHEMA_VERSION,
        "report": report,
        "params": params or {},
        "input": {"path": str(input_path), "sha256": sha256_file(Path(input_path))} if input_path else None,
        "outputs_sha256": files,
        "label_semantics": "1 = event reported for this pair in TWOSIDES; 0 = unreported (NOT proven safe)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_event_table(out_dir: Path) -> EventTable:
    out_dir = Path(out_dir)
    return EventTable(
        drugs=pd.read_csv(out_dir / "drugs.csv"),
        pairs=pd.read_csv(out_dir / "pairs.csv"),
        events=pd.read_csv(out_dir / "events_all.csv", keep_default_na=False),
        Y=sp.load_npz(out_dir / "labels.npz").tocsr(),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def read_edges_chunked(path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    """Read the 542 MB edge file with low memory: every column is stored as a
    category (a few thousand unique strings + int codes), not 4.6M strings."""
    from pandas.api.types import union_categoricals

    cols = ["source", "target", "interaction_type"]
    parts = {c: [] for c in cols}
    for ch in pd.read_csv(path, sep=",", dtype=str, chunksize=chunksize, usecols=cols):
        for c in cols:
            parts[c].append(ch[c].astype("category"))
    out = {c: pd.Series(union_categoricals(parts[c], ignore_order=True)) for c in cols}
    return pd.DataFrame(out)


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edges", required=True, type=Path, help="drug_drug_edges.csv (source,target,interaction_type)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--canonicalise", action="store_true", help="RDKit-canonicalise SMILES (reports how many changed)")
    a = ap.parse_args(list(argv) if argv is not None else None)

    edges = read_edges_chunked(a.edges)
    table, report = build_event_table(edges, canonicalise=a.canonicalise)
    manifest = save_event_table(table, a.out, report, input_path=a.edges, params={"canonicalise": a.canonicalise})
    print(json.dumps(manifest["report"], indent=2))


if __name__ == "__main__":
    main()
