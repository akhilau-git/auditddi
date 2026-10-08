"""Honest per-drug feature store (Guide 1, sections 3 and 4.3-4.5).

Blocks
------
ecfp    ECFP6 bits (radius 3, 1024)                       known for every valid SMILES
target  multi-hot over real ChEMBL mechanism targets      known only if the drug has >= 1
gene    multi-hot over real PharmGKB gene annotations     known only if the drug has >= 1

Rules enforced here
-------------------
* No imputation and no fallback target.  A drug without a real annotation gets an
  all-zero block AND has_<block> = 0, so a model can tell "no target known" from
  "target not hit".  (The older pipeline wrote CYP3A4 for drugs with no gene.)
* Nothing in this module reads event labels, so vocabularies built here cannot leak.
* Everything is keyed by the drug_idx of the event table, and the SMILES-list hash
  is stored so a store can never be paired with a different drug list.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

BLOCKS = ("ecfp", "target", "gene")


def rdkit_canonical(smiles: str) -> Optional[str]:
    from rdkit import Chem  # type: ignore

    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m is not None else None


def parse_gene_list(x) -> List[str]:
    """Accept JSON list, python-literal list, or a delimiter-separated string."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s or s in ("[]", "{}", "nan", "None"):
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(s)
            if isinstance(v, (list, tuple, set)):
                return sorted({str(g).strip() for g in v if str(g).strip()})
        except Exception:
            pass
    parts = re.split(r"[,;|\s]+", s.strip("[]{}()'\" "))
    return sorted({p.strip("'\" ") for p in parts if p.strip("'\" ")})


def _multi_hot(n_drugs: int, items_per_drug: Dict[int, Sequence[str]]):
    vocab = sorted({it for v in items_per_drug.values() for it in v})
    idx = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((n_drugs, len(vocab)), dtype=np.uint8)
    for d, items in items_per_drug.items():
        for it in items:
            M[d, idx[it]] = 1
    has = np.zeros(n_drugs, dtype=bool)
    for d, items in items_per_drug.items():
        has[d] = len(items) > 0
    return M, vocab, has


def build_target_block(n_drugs: int, targets: pd.DataFrame, human_only: bool = False):
    """targets: columns drug_idx, uniprot[, organism]."""
    t = targets.dropna(subset=["drug_idx", "uniprot"]).copy()
    if human_only and "organism" in t:
        t = t[t.organism == "Homo sapiens"]
    per: Dict[int, List[str]] = {}
    for d, g in t.groupby("drug_idx"):
        per[int(d)] = sorted(set(g.uniprot.astype(str)))
    return _multi_hot(n_drugs, per)


def build_gene_block(drugs: pd.DataFrame, profiles: pd.DataFrame, canon_fn: Callable[[str], Optional[str]] = rdkit_canonical,
                     smiles_col: str = "canonical_smiles", genes_col: str = "genes_list"):
    """Match PharmGKB profiles to drugs by canonical SMILES (exact, no fuzzy matching)."""
    canon = {canon_fn(s): int(i) for i, s in zip(drugs["drug_idx"], drugs["smiles"]) if canon_fn(s) is not None}
    per: Dict[int, List[str]] = {}
    unmatched = 0
    for s, g in zip(profiles[smiles_col], profiles[genes_col]):
        c = canon_fn(s)
        d = canon.get(c) if c is not None else None
        genes = parse_gene_list(g)
        if d is None:
            unmatched += 1
        elif genes:
            per[d] = sorted(set(per.get(d, [])) | set(genes))
    M, vocab, has = _multi_hot(len(drugs), per)
    return M, vocab, has, {"profiles": int(len(profiles)), "unmatched_profiles": int(unmatched)}


def smiles_hash(smiles: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, smiles)).encode()).hexdigest()


def build_store(drugs: pd.DataFrame, ecfp: np.ndarray, ecfp_valid: np.ndarray, targets: pd.DataFrame, profiles: pd.DataFrame,
                canon_fn: Callable[[str], Optional[str]] = rdkit_canonical, human_only_targets: bool = False) -> dict:
    n = len(drugs)
    assert ecfp.shape[0] == n
    T, tvocab, has_t = build_target_block(n, targets, human_only_targets)
    G, gvocab, has_g, gmeta = build_gene_block(drugs, profiles, canon_fn)
    return {
        "ecfp": ecfp.astype(np.uint8), "ecfp_valid": ecfp_valid.astype(bool),
        "target": T, "has_target": has_t, "target_vocab": np.array(tvocab, dtype=str),
        "gene": G, "has_gene": has_g, "gene_vocab": np.array(gvocab, dtype=str),
        "smiles_sha256": np.array(smiles_hash(drugs["smiles"].tolist())),
        "gene_match_meta": np.array(json.dumps(gmeta)),
    }


def coverage_report(store: dict) -> dict:
    ht, hg = store["has_target"], store["has_gene"]
    n = len(ht)
    return {
        "n_drugs": int(n),
        "ecfp_valid": int(store["ecfp_valid"].sum()),
        "has_target": int(ht.sum()), "has_gene": int(hg.sum()),
        "has_target_or_gene": int((ht | hg).sum()), "has_both": int((ht & hg).sum()),
        "has_neither": int((~(ht | hg)).sum()),
        "n_target_vocab": int(len(store["target_vocab"])), "n_gene_vocab": int(len(store["gene_vocab"])),
        "gene_match": json.loads(str(store["gene_match_meta"])),
        "fraction_with_any_biology": round(float((ht | hg).mean()), 4),
    }


def save_store(store: dict, out: Path, extra: Optional[dict] = None) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **store)
    rep = coverage_report(store)
    if extra:
        rep["inputs"] = extra
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2))
    return rep


def load_store(path: Path, drugs: Optional[pd.DataFrame] = None) -> dict:
    z = np.load(path, allow_pickle=False)
    store = {k: z[k] for k in z.files}
    if drugs is not None and str(store["smiles_sha256"]) != smiles_hash(drugs["smiles"].tolist()):
        raise ValueError(f"{path} was built for a different drug list")
    return store


def feature_matrix(store: dict, features: Sequence[str]) -> np.ndarray:
    """Concatenate the requested blocks.  Each biology block is followed by its
    known-flag column, so pair features also expose 'both known' (product) and
    'exactly one known' (|difference|)."""
    cols = []
    for f in features:
        if f == "ecfp":
            cols.append(store["ecfp"].astype(np.float32))
        elif f == "target":
            cols += [store["target"].astype(np.float32), store["has_target"][:, None].astype(np.float32)]
        elif f == "gene":
            cols += [store["gene"].astype(np.float32), store["has_gene"][:, None].astype(np.float32)]
        else:
            raise ValueError(f"unknown feature block: {f}")
    return np.concatenate(cols, axis=1)


def biology_known(store: dict) -> np.ndarray:
    return store["has_target"] | store["has_gene"]


def stratum_of_pairs(known: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """0 = neither drug has biology, 1 = exactly one, 2 = both."""
    return known[a].astype(int) + known[b].astype(int)
