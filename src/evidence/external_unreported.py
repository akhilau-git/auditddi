"""External validation on pairs TWOSIDES NEVER REPORTED (the redesigned external test).

    python -m src.evidence.external_unreported \
        --table .../event_table_v1 --splits .../splits_v1 --name cold_drug_seed0 \
        --feature-store .../features_v3/drug_features.npz --features ecfp esm cyp \
        --checkpoint .../multilabel_tune_acct1/best/checkpoints/auditddi_ml_cold_drug_seed0_ecfp+esm+cyp.pt \
        --ddinter .../evidence_v1/ddinter_pairs.csv --label-pairs .../evidence_v1/label_pairs.csv --out .../evidence_v1

WHY: inside TWOSIDES pairs, even the TRUE event count agrees with DDInter only at AUROC ~0.52-0.54, so that
comparison cannot validate anything.  Here the candidates are all pairs among the drugs DDInter covers that are NOT
TWOSIDES pairs.  The model never saw these pairs' labels.  Positives are the DDInter / label-mention pairs among them;
everything else is "not listed" (NOT 'safe': DDInter is incomplete, so some negatives are unknown positives).

Scores (per pair, computed from the model's event probabilities):  mean_prob, top20_mean, mean_logit.
Baseline: log1p(degree A) + log1p(degree B), with degree = partners among the split's TRAINING pairs.
Reported for all candidates and separately by REGIME (0 = both drugs seen in training, 1 = one unseen, 2 = both unseen),
because a model can only be expected to add information beyond popularity where it knows something about the drugs.
The AUROC confidence intervals resample pairs, which share drugs, so they are optimistic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.evidence.external_agreement import auroc, boot_ci, flags_for_pairs


def candidate_pairs(twosides_pairs: pd.DataFrame, ddinter: pd.DataFrame):
    """All unordered pairs among drugs that occur in DDInter, minus pairs that are TWOSIDES pairs."""
    scope = np.array(sorted(set(ddinter.drug_a.astype(int)) | set(ddinter.drug_b.astype(int))), dtype=np.int64)
    ii, jj = np.triu_indices(len(scope), 1)
    a, b = scope[ii], scope[jj]
    two = np.minimum(twosides_pairs.drug_a, twosides_pairs.drug_b).astype(np.int64) * 100000 + np.maximum(twosides_pairs.drug_a, twosides_pairs.drug_b)
    keep = ~np.isin(a * 100000 + b, two.to_numpy())
    return a[keep], b[keep]


def popularity(n_drugs: int, table_a: np.ndarray, table_b: np.ndarray, train_rows: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    deg = np.bincount(np.concatenate([table_a[train_rows], table_b[train_rows]]), minlength=n_drugs).astype(float)
    return np.log1p(deg[a]) + np.log1p(deg[b])


def evaluate_candidates(scores: Dict[str, np.ndarray], a: np.ndarray, b: np.ndarray, regime: np.ndarray, ddinter: pd.DataFrame,
                        label_pairs: Optional[pd.DataFrame], n_boot: int = 100, min_pos: int = 30) -> pd.DataFrame:
    flags = flags_for_pairs(a, b, ddinter, label_pairs)
    subsets = {"all": np.ones(len(a), bool), "regime0_both_seen": regime == 0, "regime1_one_unseen": regime == 1, "regime2_both_unseen": regime == 2}
    rows = []
    for fname, y in flags.items():
        for sub, m in subsets.items():
            ym = y[m]
            n_pos = int(ym.sum())
            for sname, s in scores.items():
                sm = s[m]
                const = len(sm) == 0 or float(np.ptp(sm)) == 0.0
                row = {"flag": fname, "subset": sub, "score": sname, "n_pairs": int(m.sum()), "n_flag_pos": n_pos,
                       "flag_prevalence": float(ym.mean()) if len(ym) else float("nan"), "score_is_constant": bool(const)}
                if n_pos < min_pos or n_pos > len(ym) - min_pos or const:
                    row.update({"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
                else:
                    lo, hi = boot_ci(ym, sm, n_boot)
                    row.update({"auroc": auroc(ym, sm), "ci_low": lo, "ci_high": hi})
                rows.append(row)
    return pd.DataFrame(rows)


def score_candidates(model, F, a: np.ndarray, b: np.ndarray, batch: int = 4096, k: int = 20, device=None) -> Dict[str, np.ndarray]:
    """Pair-level summaries of the model's event probabilities, computed batch by batch (no n_pairs x n_events matrix)."""
    import torch

    dev = device or F.device
    mp, tk, ml = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(a), batch):
            ai = torch.tensor(a[i:i + batch], dtype=torch.long, device=dev)
            bi = torch.tensor(b[i:i + batch], dtype=torch.long, device=dev)
            logits = model(F[ai], F[bi])
            p = torch.sigmoid(logits)
            mp.append(p.mean(1).cpu().numpy())
            tk.append(p.topk(min(k, p.shape[1]), dim=1).values.mean(1).cpu().numpy())
            ml.append(logits.mean(1).cpu().numpy())
    return {"mean_prob": np.concatenate(mp), "top20_mean": np.concatenate(tk), "mean_logit": np.concatenate(ml)}


def main(argv: Optional[Sequence[str]] = None) -> pd.DataFrame:
    import torch

    from src.data_prep.build_event_table import load_event_table
    from src.features.drug_features import feature_layout, feature_matrix, load_store
    from src.multilabel.model import load_checkpoint
    from src.trust.regime import regime_of_pairs, seen_mask

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--feature-store", required=True, type=Path)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--ddinter", required=True, type=Path)
    ap.add_argument("--label-pairs", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)

    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    table = load_event_table(a.table)
    store = load_store(a.feature_store, table.drugs)
    layout, _ = feature_layout(store, a.features)
    model, meta = load_checkpoint(a.checkpoint, str(dev))
    info = json.loads((a.splits / f"{a.name}.json").read_text())
    if meta["features"] != list(a.features):
        raise ValueError(f"checkpoint was trained on features {meta['features']}, not {a.features}")
    if meta["vocab_sha256"] != info["vocab"]["sha256"] or meta["split_digest"] != info["sha256_partition_digest"]:
        raise ValueError("checkpoint does not belong to this split / vocabulary (hash mismatch)")
    if {k: (v["start"], v["end"], v["flag"]) for k, v in model.layout.items()} != {k: (v["start"], v["end"], v["flag"]) for k, v in layout.items()}:
        raise ValueError("the feature store layout differs from the one the checkpoint was trained with")

    A, B = table.pairs.drug_a.to_numpy(), table.pairs.drug_b.to_numpy()
    split = np.load(a.splits / f"{a.name}.npz")
    dd = pd.read_csv(a.ddinter)
    lp = pd.read_csv(a.label_pairs) if a.label_pairs and a.label_pairs.exists() else None
    ca, cb = candidate_pairs(table.pairs, dd)
    F = torch.tensor(feature_matrix(store, a.features), dtype=torch.float32, device=dev)
    scores = score_candidates(model, F, ca, cb, device=dev)
    scores["popularity_baseline"] = popularity(len(table.drugs), A, B, split["train"], ca, cb)
    regime = regime_of_pairs(seen_mask(len(table.drugs), A, B, split["train"]), ca, cb)
    df = evaluate_candidates(scores, ca, cb, regime, dd, lp, a.n_boot)
    df.insert(0, "split", a.name)
    df.insert(1, "features", "+".join(a.features))
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / "external_unreported.csv"
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    pd.concat([prev, df], ignore_index=True).to_csv(out, index=False)
    print(f"candidates: {len(ca)} unreported pairs among {len(set(ca) | set(cb))} drugs; regimes {np.bincount(regime, minlength=3).tolist()}")
    show = df[(df.flag == "ddinter_moderate_major") & df.score.isin(["mean_logit", "popularity_baseline"])]
    print(show.pivot_table(index="subset", columns="score", values="auroc").round(3).to_string())
    return df


if __name__ == "__main__":
    main()
