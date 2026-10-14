"""Split protocol for multi-label DDI (Guide 1, sections 5.2 and 5.3).

Every row of the event table is one unordered drug pair that carries its COMPLETE
event set, so a pair can never be split across partitions.

Regimes
-------
transductive   Both drugs of every val/test pair occur in training; only the exact
               pair is unseen.                                  (Guide 1 split C)
cold_drug      Drugs are partitioned into train/val/test drug sets.
                 train pairs : both drugs in the train set
                 *_s2 pairs  : exactly one drug unseen, the other in the train set
                 *_s1 pairs  : both drugs unseen (same held-out set)
               Pairs that mix val and test drugs are dropped and counted.
                                                                (Guide 1 split B)
scaffold       Same as cold_drug but whole Bemis-Murcko scaffold groups are assigned
               to val/test, so no val/test scaffold occurs in training.
                                                                (Guide 1 split D)
random_pair    Historical comparison only.                      (Guide 1 split A)

``audit_split`` verifies the guarantees and ``assert_clean`` raises on violation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

TRAIN, VAL, TEST = 0, 1, 2


# --------------------------------------------------------------------------- #
# container
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    kind: str
    seed: int
    train: np.ndarray
    val: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]
    dropped: np.ndarray
    drug_role: Optional[np.ndarray] = None         # per drug: 0 train / 1 val / 2 test
    meta: dict = field(default_factory=dict)

    def sizes(self) -> dict:
        return {
            "train": int(len(self.train)),
            "val": {k: int(len(v)) for k, v in self.val.items()},
            "test": {k: int(len(v)) for k, v in self.test.items()},
            "dropped": int(len(self.dropped)),
        }

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.kind.encode())
        h.update(str(self.seed).encode())
        for name, arr in [("train", self.train)] + [(f"val_{k}", v) for k, v in sorted(self.val.items())] + [
            (f"test_{k}", v) for k, v in sorted(self.test.items())
        ]:
            h.update(name.encode())
            h.update(np.ascontiguousarray(np.sort(arr).astype(np.int64)).tobytes())
        return h.hexdigest()


def _pairs_arrays(pairs: pd.DataFrame):
    return pairs["drug_a"].to_numpy(dtype=np.int64), pairs["drug_b"].to_numpy(dtype=np.int64)


def _empty() -> np.ndarray:
    return np.array([], dtype=np.int64)


# --------------------------------------------------------------------------- #
# split A: random pair (historical)
# --------------------------------------------------------------------------- #
def random_pair_split(pairs: pd.DataFrame, frac_val: float = 0.1, frac_test: float = 0.2, seed: int = 0) -> SplitResult:
    n = len(pairs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test, n_val = int(round(n * frac_test)), int(round(n * frac_val))
    test, val, train = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    return SplitResult("random_pair", seed, np.sort(train), {"all": np.sort(val)}, {"all": np.sort(test)}, _empty())


# --------------------------------------------------------------------------- #
# split C: transductive (both drugs seen)
# --------------------------------------------------------------------------- #
def transductive_split(pairs: pd.DataFrame, frac_val: float = 0.1, frac_test: float = 0.2, seed: int = 0) -> SplitResult:
    """Random pair split, then move any held-out pair that contains a drug absent
    from training back into training.  Sequential, so it moves the minimum."""
    a, b = _pairs_arrays(pairs)
    base = random_pair_split(pairs, frac_val, frac_test, seed)
    train_set = np.zeros(int(max(a.max(), b.max())) + 1, dtype=bool)
    train_set[a[base.train]] = True
    train_set[b[base.train]] = True

    moved = []
    keep = {"val": [], "test": []}
    for name, idx in (("val", base.val["all"]), ("test", base.test["all"])):
        for p in idx:
            if train_set[a[p]] and train_set[b[p]]:
                keep[name].append(p)
            else:
                moved.append(p)
                train_set[a[p]] = True
                train_set[b[p]] = True
    train = np.sort(np.concatenate([base.train, np.array(moved, dtype=np.int64)]))
    res = SplitResult(
        "transductive", seed, train,
        {"all": np.array(sorted(keep["val"]), dtype=np.int64)},
        {"all": np.array(sorted(keep["test"]), dtype=np.int64)},
        _empty(),
    )
    res.meta["pairs_moved_back_to_train"] = int(len(moved))
    return res


# --------------------------------------------------------------------------- #
# drug-partition splits (cold_drug and scaffold)
# --------------------------------------------------------------------------- #
def _split_from_roles(pairs: pd.DataFrame, role: np.ndarray, kind: str, seed: int) -> SplitResult:
    a, b = _pairs_arrays(pairs)
    ra, rb = role[a], role[b]
    idx = np.arange(len(pairs), dtype=np.int64)

    train = idx[(ra == TRAIN) & (rb == TRAIN)]
    val_s2 = idx[((ra == VAL) & (rb == TRAIN)) | ((ra == TRAIN) & (rb == VAL))]
    val_s1 = idx[(ra == VAL) & (rb == VAL)]
    test_s2 = idx[((ra == TEST) & (rb == TRAIN)) | ((ra == TRAIN) & (rb == TEST))]
    test_s1 = idx[(ra == TEST) & (rb == TEST)]
    used = np.zeros(len(pairs), dtype=bool)
    for arr in (train, val_s2, val_s1, test_s2, test_s1):
        used[arr] = True
    dropped = idx[~used]
    return SplitResult(
        kind, seed, train,
        {"s2": val_s2, "s1": val_s1, "all": np.sort(np.concatenate([val_s2, val_s1]))},
        {"s2": test_s2, "s1": test_s1},
        dropped, drug_role=role,
    )


def cold_drug_split(pairs: pd.DataFrame, n_drugs: int, frac_val_drugs: float = 0.10, frac_test_drugs: float = 0.20, seed: int = 0) -> SplitResult:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_drugs)
    n_test, n_val = int(round(n_drugs * frac_test_drugs)), int(round(n_drugs * frac_val_drugs))
    role = np.full(n_drugs, TRAIN, dtype=np.int8)
    role[perm[:n_test]] = TEST
    role[perm[n_test:n_test + n_val]] = VAL
    return _split_from_roles(pairs, role, "cold_drug", seed)


def scaffold_of(smiles: str) -> str:
    """Bemis-Murcko scaffold of the largest fragment.  Needs RDKit.  Acyclic
    molecules return '' (their own group)."""
    from rdkit import Chem  # type: ignore
    from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    frags = Chem.GetMolFrags(mol, asMols=True)
    big = max(frags, key=lambda m: m.GetNumAtoms())
    return MurckoScaffold.MurckoScaffoldSmiles(mol=big, includeChirality=False)


def scaffold_split(
    pairs: pd.DataFrame,
    drugs: pd.DataFrame,
    frac_val_drugs: float = 0.10,
    frac_test_drugs: float = 0.20,
    seed: int = 0,
    scaffold_fn: Callable[[str], str] = scaffold_of,
) -> SplitResult:
    """Assign whole scaffold groups to test, then val; everything else is train.

    Groups are visited in a seeded random order.  A group is added to the held-out
    set only if it fits in the remaining quota, so the drug fractions are respected
    (unlike a 'largest first' fill, which would put all rare scaffolds in test)."""
    n_drugs = len(drugs)
    scaf = [scaffold_fn(s) for s in drugs["smiles"]]
    groups: Dict[str, list] = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s, []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    order = [keys[i] for i in rng.permutation(len(keys))]

    n_test, n_val = int(round(n_drugs * frac_test_drugs)), int(round(n_drugs * frac_val_drugs))
    role = np.full(n_drugs, TRAIN, dtype=np.int8)
    got_test = got_val = 0
    for k in order:
        g = groups[k]
        if got_test + len(g) <= n_test:
            role[g] = TEST
            got_test += len(g)
        elif got_val + len(g) <= n_val:
            role[g] = VAL
            got_val += len(g)
    res = _split_from_roles(pairs, role, "scaffold", seed)
    res.meta.update(
        {
            "n_scaffold_groups": len(groups),
            "test_drugs": int(got_test),
            "val_drugs": int(got_val),
            "scaffold_of_drug": scaf,
        }
    )
    return res


# --------------------------------------------------------------------------- #
# audit  (Guide 1 section 5.3)
# --------------------------------------------------------------------------- #
def audit_split(res: SplitResult, pairs: pd.DataFrame) -> dict:
    a, b = _pairs_arrays(pairs)
    parts = {"train": res.train}
    parts.update({f"val_{k}": v for k, v in res.val.items() if k != "all"} if "s2" in res.val else {"val_all": res.val["all"]})
    parts.update({f"test_{k}": v for k, v in res.test.items()})

    checks: dict = {}
    seen = np.zeros(len(pairs), dtype=np.int32)
    for v in parts.values():
        seen[v] += 1
    seen[res.dropped] += 1
    checks["every_pair_in_exactly_one_partition_or_dropped"] = bool((seen == 1).all())

    def drugs_of(idx):
        return set(a[idx].tolist()) | set(b[idx].tolist())

    train_drugs = drugs_of(res.train)
    held = [k for k in parts if k != "train"]

    if res.kind in ("transductive", "random_pair"):
        if res.kind == "transductive":
            checks["all_val_test_drugs_seen_in_train"] = all(drugs_of(parts[k]) <= train_drugs for k in held)
    else:
        for k in held:
            idx = parts[k]
            if len(idx) == 0:
                continue
            ra, rb = res.drug_role[a[idx]], res.drug_role[b[idx]]
            n_unseen = (ra != TRAIN).astype(int) + (rb != TRAIN).astype(int)
            want = 2 if k.endswith("s1") else 1
            checks[f"{k}_has_{want}_unseen_drug"] = bool((n_unseen == want).all())
        checks["train_drugs_disjoint_from_held_out_drugs"] = bool(
            not (train_drugs & set(np.where(res.drug_role != TRAIN)[0].tolist()))
        )

    if res.kind == "scaffold":
        scaf = res.meta["scaffold_of_drug"]
        tr_sc = {scaf[d] for d in train_drugs}
        held_drugs = np.where(res.drug_role != TRAIN)[0]
        held_sc = {scaf[d] for d in held_drugs}
        checks["no_scaffold_overlap_train_vs_held_out"] = bool(not (tr_sc & held_sc))

    checks["train_nonempty"] = bool(len(res.train) > 0)
    return checks


def assert_clean(res: SplitResult, pairs: pd.DataFrame) -> dict:
    checks = audit_split(res, pairs)
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise AssertionError(f"split '{res.kind}' seed={res.seed} failed audit: {bad}")
    return checks


# --------------------------------------------------------------------------- #
# event support: metrics are undefined for events with ~no test positives
# --------------------------------------------------------------------------- #
def event_support(Y, vocab: pd.DataFrame, partitions: Dict[str, np.ndarray], min_pos: int = 5) -> pd.DataFrame:
    cols = vocab["event_idx"].to_numpy(dtype=np.int64)
    out = vocab[["head_idx", "event"]].copy()
    for name, idx in partitions.items():
        pos = np.asarray(Y[np.asarray(idx, dtype=np.int64)][:, cols].sum(axis=0)).ravel() if len(idx) else np.zeros(len(cols))
        out[f"pos_{name}"] = pos.astype(np.int64)
        out[f"evaluable_{name}"] = (pos >= min_pos) & ((len(idx) - pos) >= min_pos)
    return out


def scaled_min_pairs(min_pairs_full: int, n_train: int, n_total: int) -> int:
    """Guide 1's ~863-event rule is 'at least 750 pairs' on the full 63,473-pair
    table.  On a training partition holding a fraction f of the pairs the same
    relative frequency is 750 * f (rounded up)."""
    return int(np.ceil(min_pairs_full * n_train / max(n_total, 1)))


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_split(res: SplitResult, out_dir: Path, name: str, audit: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"train": res.train}
    arrays.update({f"val_{k}": v for k, v in res.val.items()})
    arrays.update({f"test_{k}": v for k, v in res.test.items()})
    arrays["dropped"] = res.dropped
    if res.drug_role is not None:
        arrays["drug_role"] = res.drug_role
    np.savez_compressed(out_dir / f"{name}.npz", **arrays)
    meta = {k: v for k, v in res.meta.items() if k != "scaffold_of_drug"}
    info = {
        "name": name,
        "kind": res.kind,
        "seed": res.seed,
        "sizes": res.sizes(),
        "sha256_partition_digest": res.digest(),
        "audit": audit,
        "meta": meta,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(info, indent=2))
    return info


def load_split(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}
