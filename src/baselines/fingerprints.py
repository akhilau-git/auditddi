"""ECFP6 drug fingerprints and pair features (Guide 1, sections 4.2, 4.6, 4.13).

ECFP6 = Morgan radius 3, 1024 bits.  (The earlier repo code used radius 2 = ECFP4.)
"""
from __future__ import annotations

import hashlib
from typing import Sequence, Tuple

import numpy as np

RADIUS = 3
N_BITS = 1024


def ecfp6_matrix(smiles: Sequence[str], radius: int = RADIUS, n_bits: int = N_BITS) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bits[n, n_bits] uint8, valid[n] bool).  Invalid SMILES -> zero row,
    valid=False, so the caller can report the count instead of hiding it."""
    from rdkit import Chem  # type: ignore

    try:
        from rdkit.Chem import rdFingerprintGenerator  # type: ignore

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

        def fp(m):
            return gen.GetFingerprintAsNumPy(m).astype(np.uint8)

    except Exception:  # older RDKit
        from rdkit import DataStructs  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        def fp(m):
            v = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(v, arr)
            return arr

    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    ok = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            X[i] = fp(m)
            ok[i] = True
    return X, ok


def smiles_list_hash(smiles: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, smiles)).encode()).hexdigest()


def pair_features(F: np.ndarray, a: np.ndarray, b: np.ndarray, mode: str = "symmetric") -> np.ndarray:
    """Pair representation from per-drug features F[n_drugs, d].

    symmetric : [F_a + F_b || |F_a - F_b| || F_a * F_b]   f(A,B) == f(B,A) exactly
    concat    : [F_a || F_b]                                order-dependent (ablation)
    """
    Fa = F[a].astype(np.float32)
    Fb = F[b].astype(np.float32)
    if mode == "symmetric":
        return np.concatenate([Fa + Fb, np.abs(Fa - Fb), Fa * Fb], axis=1)
    if mode == "concat":
        return np.concatenate([Fa, Fb], axis=1)
    raise ValueError(f"unknown pair mode: {mode}")
