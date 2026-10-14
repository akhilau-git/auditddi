"""Train the multi-label AuditDDI model on one audited split.

    python -m src.multilabel.train \
        --table /content/drive/MyDrive/auditddi-data/event_table_v1 \
        --splits /content/drive/MyDrive/auditddi-data/splits_v1 --name cold_drug_seed0 \
        --feature-store /content/drive/MyDrive/auditddi-data/features_v3/drug_features.npz \
        --features ecfp esm cyp --out /content/drive/MyDrive/auditddi-data/multilabel_acct1 --save-scores

Everything the baselines use is reused: the event table, the audited split and its FROZEN vocabulary, the
per-event pos_weight (computed from training pairs only), the metric code, and the score-file format, so
src.trust.run_calibration / selective_analysis and src.evidence.external_agreement work on the output with
`--models auditddi_ml` (and `--feature-tag <features joined by +>`).

Loss: BCE-with-logits per event, weighted by pos_weight_e = negatives_e / positives_e (training pairs only).
Optimiser: AdamW, gradient clipping 1.0, ReduceLROnPlateau on validation macro AUPRC, early stopping.
Validation pairs are used only for early stopping and the decision threshold; test pairs are never touched
before the final evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.baselines.models import monitor_score
from src.baselines.run_baselines import append_results, drop_matching, partitions_of
from src.data_prep.build_event_table import labels_for, load_event_table
from src.evaluation.multilabel_metrics import best_global_threshold, evaluate
from src.features.drug_features import biology_known, feature_layout, feature_matrix, load_store, stratum_of_pairs
from src.multilabel.model import AuditDDIML, drop_biology, predict_proba, save_checkpoint

MODEL_NAME = "auditddi_ml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--feature-store", required=True, type=Path)
    ap.add_argument("--features", nargs="+", default=["ecfp"], choices=["ecfp", "target", "gene", "tclass", "atc", "cyp", "esm"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--tower-hidden", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--bio-dropout", type=float, default=0.15, help="per-drug probability of hiding all biology during training")
    ap.add_argument("--pos-weight-clip", type=float, default=None)
    ap.add_argument("--model-seed", type=int, default=None, help="default: the split's seed")
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--val-only", action="store_true", help="evaluate ONLY the validation partition (used for hyper-parameter selection; test pairs are never scored)")
    ap.add_argument("--save-scores", action="store_true")
    ap.add_argument("--no-checkpoint", action="store_true")
    a = ap.parse_args(list(argv) if argv is not None else None)

    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    table = load_event_table(a.table)
    store = load_store(a.feature_store, table.drugs)
    F_np = feature_matrix(store, a.features)
    layout, _ = feature_layout(store, a.features)
    known = biology_known(store)
    split = {k: v for k, v in np.load(a.splits / f"{a.name}.npz").items()}
    info = json.loads((a.splits / f"{a.name}.json").read_text())
    vocab = pd.read_csv(a.splits / f"{a.name}_vocab.csv", keep_default_na=False)
    if len(vocab) == 0:
        raise ValueError("empty event vocabulary; regenerate the splits with a lower --min-pairs-full")
    flabel = "+".join(a.features)
    res_path = a.out / "results.csv"
    if res_path.exists() and not a.force and not a.val_only:
        prev = pd.read_csv(res_path)
        if "features" in prev and ((prev["split"] == a.name) & (prev["model"] == MODEL_NAME) & (prev["features"] == flabel)).any():
            print(f"skip {a.name} {MODEL_NAME} {flabel} (already in results.csv; use --force)")
            return {"skipped": True}

    parts = partitions_of(split)
    if a.val_only:
        parts = {"val": parts["val"]}
    tr = split["train"]
    A_np, B_np = table.pairs["drug_a"].to_numpy(), table.pairs["drug_b"].to_numpy()
    Ytr = labels_for(table.Y, tr, vocab).astype(np.float32)
    Yparts = {k: labels_for(table.Y, v, vocab).astype(np.float32) for k, v in parts.items()}
    pw = vocab["pos_weight"].to_numpy(np.float32) if "pos_weight" in vocab else ((len(tr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1.0)).astype(np.float32)
    if a.pos_weight_clip is not None:
        pw = np.minimum(pw, a.pos_weight_clip)

    seed = info["seed"] if a.model_seed is None else a.model_seed
    set_seed(seed)
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    F = torch.tensor(F_np, dtype=torch.float32, device=dev)
    A_t, B_t = torch.tensor(A_np, dtype=torch.long, device=dev), torch.tensor(B_np, dtype=torch.long, device=dev)
    tr_t = torch.tensor(tr, dtype=torch.long, device=dev)
    Ytr_t = torch.tensor(Ytr, device=dev)
    part_t = {k: torch.tensor(v, dtype=torch.long, device=dev) for k, v in parts.items()}

    model = AuditDDIML(layout, len(vocab), a.emb, a.tower_hidden, a.hidden, a.dropout).to(dev)
    model.init_output_bias((Ytr.sum(0) + 1.0) / (len(Ytr) + 2.0))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
    gen = torch.Generator()
    gen.manual_seed(seed)

    best, best_state, bad, history = -np.inf, None, 0, []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=gen).to(dev)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), a.batch_size):
            idx = perm[i:i + a.batch_size]
            if len(idx) < 2:                                   # BatchNorm needs more than one row
                continue
            rows = tr_t[idx]
            xa = drop_biology(F[A_t[rows]], layout, a.bio_dropout)
            xb = drop_biology(F[B_t[rows]], layout, a.bio_dropout)
            loss = loss_fn(model(xa, xb), Ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
            nb += 1
        v = part_t["val"]
        Sv = predict_proba(model, F, A_t[v], B_t[v])
        score = monitor_score(Yparts["val"], Sv, "auprc", a.min_pos)
        score = -np.inf if not np.isfinite(score) else float(score)
        sched.step(score if np.isfinite(score) else 0.0)
        history.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_auprc": score, "lr": opt.param_groups[0]["lr"]})
        if score > best + 1e-5:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    scores = {k: predict_proba(model, F, A_t[t], B_t[t]) for k, t in part_t.items()}
    thr = best_global_threshold(Yparts["val"], scores["val"])
    base = {"split": a.name, "kind": info["kind"], "seed": info["seed"], "model": MODEL_NAME, "pair_mode": "symmetric", "features": flabel,
            "scenario": "structure_only" if a.features == ["ecfp"] else "annotation_rich", "stratified": True, "n_train": int(len(tr)),
            "n_heads": int(len(vocab)), "train_seconds": round(time.time() - t0, 1), "epochs_run": len(history), "model_seed": seed,
            "vocab_sha256": info["vocab"]["sha256"], "split_digest": info["sha256_partition_digest"]}
    rows = []
    for k, rr in parts.items():
        rows.append({**base, "partition": k, **evaluate(Yparts[k].astype(np.int8), scores[k], threshold=thr, min_pos=a.min_pos)})
        if k != "val":
            st = stratum_of_pairs(known, A_np[rr], B_np[rr])
            for sv in (0, 1, 2):
                sel = np.where(st == sv)[0]
                if len(sel) >= 50:
                    rows.append({**{x: base[x] for x in ("split", "kind", "seed", "model", "pair_mode", "features", "scenario", "stratified", "n_train", "n_heads", "vocab_sha256", "split_digest")},
                                 "partition": f"{k}|bio{sv}", **evaluate(Yparts[k][sel].astype(np.int8), scores[k][sel], threshold=thr, min_pos=a.min_pos, with_calibration=False)})
        if a.save_scores:
            tag = "" if flabel == "ecfp" else "@" + flabel
            np.savez_compressed(a.out / f"scores_{a.name}_{MODEL_NAME}{tag}_symmetric_{k}.npz", S=scores[k].astype(np.float16), rows=rr)
    if a.force:
        drop_matching(res_path, a.name, MODEL_NAME, "symmetric", flabel, True)
    if not a.val_only:
        append_results(res_path, rows)
    if not a.no_checkpoint and not a.val_only:
        meta = {"features": a.features, "split": a.name, "seed": seed, "vocab_sha256": info["vocab"]["sha256"], "split_digest": info["sha256_partition_digest"],
                "feature_store_sha256": sha256_file(a.feature_store), "events": vocab["event"].tolist(), "threshold": thr, "history": history, "args": {k: str(v) for k, v in vars(a).items()}}
        save_checkpoint(a.out / "checkpoints" / f"{MODEL_NAME}_{a.name}_{flabel}.pt", model, meta)
    t = [r for r in rows if r["partition"].startswith("test") and "|" not in r["partition"]]
    print(f"{a.name:20s} {MODEL_NAME} {flabel:20s} " + " | ".join(f"{r['partition']}: AUROC {r['macro_auroc']:.3f} AUPRC {r['macro_auprc']:.3f} (n={r['n_events_evaluable']})" for r in t))
    return {"best_val_auprc": float(best), "epochs": len(history), "val_macro_auroc": next((r["macro_auroc"] for r in rows if r["partition"] == "val"), float("nan"))}


if __name__ == "__main__":
    main()
