"""Train and evaluate every baseline on every audited split.

    python -m src.baselines.run_baselines \
        --table  /content/drive/MyDrive/auditddi-data/event_table_v1 \
        --splits /content/drive/MyDrive/auditddi-data/splits_v1 \
        --out    /content/drive/MyDrive/auditddi-data/baselines_v1 \
        --names cold_drug_seed0 scaffold_seed0 transductive_seed0 \
        --models prior logreg mlp512 rf_pca

Results are APPENDED to results.csv after every (split, model, pair-mode) run and
finished runs are skipped on restart, so a Colab disconnect costs at most one run.
The decision threshold for set-valued metrics is chosen on the VALIDATION
partition only.  Model weights never see validation/test labels except through
early stopping on validation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.baselines.fingerprints import N_BITS, RADIUS, ecfp6_matrix, pair_features, smiles_list_hash
from src.baselines.models import make_model
from src.data_prep.build_event_table import labels_for, load_event_table
from src.evaluation.multilabel_metrics import best_global_threshold, evaluate
from src.features.drug_features import biology_known, feature_matrix, load_store, stratum_of_pairs


def load_fingerprints(drugs: pd.DataFrame, path: Path) -> np.ndarray:
    """Cache ECFP6 bits; refuse a cache built for a different drug list."""
    smi = drugs["smiles"].tolist()
    key = smiles_list_hash(smi)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["smiles_sha256"]) != key:
            raise ValueError(f"{path} was built for a different drug list; delete it or use another path")
        return z["bits"]
    bits, ok = ecfp6_matrix(smi, RADIUS, N_BITS)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, bits=bits, valid=ok, smiles_sha256=np.array(key), radius=RADIUS, n_bits=N_BITS)
    if not ok.all():
        print(f"WARNING: {(~ok).sum()} drugs have invalid SMILES and get all-zero fingerprints: {np.where(~ok)[0].tolist()}")
    return bits


def append_results(path: Path, rows: list) -> None:
    """Merge new rows into results.csv even if their columns differ from earlier
    rows, and write atomically so a disconnect cannot leave a half-written file."""
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True, sort=False)
    tmp = path.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)


def drop_matching(path: Path, split: str, model: str, mode: str, features: str, stratified: bool) -> int:
    """Remove earlier result rows of exactly this run (used by --force) so a rerun
    REPLACES them instead of duplicating.  Returns how many rows were removed."""
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    f = df["features"].fillna("ecfp") if "features" in df else pd.Series("ecfp", index=df.index)
    st = df["stratified"].eq(True) if "stratified" in df else pd.Series(False, index=df.index)
    m = (df["split"] == split) & (df["model"] == model) & (df["pair_mode"] == mode) & (f == features) & (st == stratified)
    if m.any():
        tmp = path.with_suffix(".csv.tmp")
        df[~m].to_csv(tmp, index=False)
        tmp.replace(path)
    return int(m.sum())


def partitions_of(split: dict) -> dict:
    parts = {}
    if "val_all" in split:
        parts["val"] = split["val_all"]
    for k, v in split.items():
        if k.startswith("test_"):
            parts[k] = v
    return parts


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--splits", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--names", nargs="*", default=None, help="split names, e.g. cold_drug_seed0 (default: all in manifest)")
    ap.add_argument("--models", nargs="+", default=["prior", "logreg", "mlp512"])
    ap.add_argument("--pair-modes", nargs="+", default=["symmetric"], choices=["symmetric", "concat"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fingerprints", type=Path, default=None, help="ECFP6 cache (.npz); default <out>/ecfp6_r3_1024.npz")
    ap.add_argument("--feature-store", type=Path, default=None, help="drug_features.npz from src.features.drug_features")
    ap.add_argument("--features", nargs="+", default=["ecfp"], choices=["ecfp", "target", "gene"],
                    help="blocks to concatenate; with --feature-store, results are also stratified by biology availability")
    ap.add_argument("--force", action="store_true", help="retrain even if this run is already in results.csv; its old rows are replaced")
    ap.add_argument("--save-scores", action="store_true", help="store float16 test scores for paired tests later")
    ap.add_argument("--min-pos", type=int, default=5)
    a = ap.parse_args(list(argv) if argv is not None else None)

    a.out.mkdir(parents=True, exist_ok=True)
    table = load_event_table(a.table)
    if a.feature_store is not None:
        store = load_store(a.feature_store, table.drugs)
        F = feature_matrix(store, a.features)
        known = biology_known(store)
        print(f"feature store: blocks={a.features} dim={F.shape[1]} drugs with any real biology: {int(known.sum())}/{len(known)}")
    else:
        if a.features != ["ecfp"]:
            raise SystemExit("--features other than ecfp need --feature-store")
        F = load_fingerprints(table.drugs, a.fingerprints or a.out / "ecfp6_r3_1024.npz").astype(np.float32)
        known = None
    flabel = "+".join(a.features)
    stratified = known is not None
    A = table.pairs["drug_a"].to_numpy()
    B = table.pairs["drug_b"].to_numpy()

    names = a.names or [s["name"] for s in json.loads((a.splits / "splits_manifest.json").read_text())["splits"]]
    res_path = a.out / "results.csv"
    done = set()
    if res_path.exists():
        prev = pd.read_csv(res_path)
        if "features" not in prev:
            prev["features"] = "ecfp"
        if "stratified" not in prev:
            prev["stratified"] = False
        prev["features"] = prev["features"].fillna("ecfp")
        prev["stratified"] = prev["stratified"].eq(True)
        done = set(zip(prev["split"], prev["model"], prev["pair_mode"], prev["features"], prev["stratified"]))

    for name in names:
        split = {k: v for k, v in np.load(a.splits / f"{name}.npz").items()}
        info = json.loads((a.splits / f"{name}.json").read_text())
        vocab = pd.read_csv(a.splits / f"{name}_vocab.csv", keep_default_na=False)
        if len(vocab) == 0:
            raise ValueError(f"{name}: empty event vocabulary; regenerate the splits with a lower --min-pairs-full")
        parts = partitions_of(split)
        tr = split["train"]
        Ytr = labels_for(table.Y, tr, vocab)
        Yparts = {k: labels_for(table.Y, v, vocab) for k, v in parts.items()}

        for mode in a.pair_modes:
            Xtr = pair_features(F, A[tr], B[tr], mode)
            Xparts = {k: pair_features(F, A[v], B[v], mode) for k, v in parts.items()}
            for mname in a.models:
                if (name, mname, mode, flabel, stratified) in done and not a.force:
                    print(f"skip {name} {mname} {mode} {flabel}")
                    continue
                t0 = time.time()
                kw = {}
                if mname not in ("prior", "rf_pca"):
                    kw = dict(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr)
                model = make_model(mname, seed=info["seed"], **kw)
                model.fit(Xtr, Ytr, Xparts["val"], Yparts["val"])
                scores = {k: model.predict_proba(X) for k, X in Xparts.items()}
                thr = best_global_threshold(Yparts["val"], scores["val"])
                rows = []
                for k in parts:
                    m = evaluate(Yparts[k].astype(np.int8), scores[k], threshold=thr, min_pos=a.min_pos)
                    row = {"split": name, "kind": info["kind"], "seed": info["seed"], "model": mname, "pair_mode": mode,
                           "features": flabel, "stratified": stratified, "partition": k, "n_train": int(len(tr)), "n_heads": int(len(vocab)), "train_seconds": round(time.time() - t0, 1),
                           "epochs_run": len(getattr(model, "history_", [])), "vocab_sha256": info["vocab"]["sha256"],
                           "split_digest": info["sha256_partition_digest"]}
                    if mode == "concat" and k != "val":
                        swapped = model.predict_proba(pair_features(F, B[parts[k]], A[parts[k]], mode))
                        row["order_sensitivity_mean_abs"] = float(np.abs(swapped - scores[k]).mean())
                    row.update(m)
                    rows.append(row)
                    if stratified and k != "val":
                        st = stratum_of_pairs(known, A[parts[k]], B[parts[k]])
                        for sv in (0, 1, 2):
                            sel = np.where(st == sv)[0]
                            if len(sel) < 50:
                                continue
                            ms = evaluate(Yparts[k][sel].astype(np.int8), scores[k][sel], threshold=thr, min_pos=a.min_pos, with_calibration=False)
                            rows.append({**{kk: vv for kk, vv in row.items() if kk in ("split", "kind", "seed", "model", "pair_mode", "features", "stratified", "n_train", "n_heads", "vocab_sha256", "split_digest")},
                                         "partition": f"{k}|bio{sv}", **ms})
                    if a.save_scores:
                        tag = "" if flabel == "ecfp" else "@" + flabel
                        np.savez_compressed(a.out / f"scores_{name}_{mname}{tag}_{mode}_{k}.npz", S=scores[k].astype(np.float16), rows=parts[k])
                if a.force:
                    n_old = drop_matching(res_path, name, mname, mode, flabel, stratified)
                    if n_old:
                        print(f"--force: replaced {n_old} earlier rows of {name} {mname} {mode} {flabel}")
                append_results(res_path, rows)
                t = [r for r in rows if r["partition"].startswith("test") and "|" not in r["partition"]]
                print(f"{name:20s} {mname:8s} {mode:9s} {flabel:18s} " + " | ".join(f"{r['partition']}: AUROC {r['macro_auroc']:.3f} AUPRC {r['macro_auprc']:.3f} (n={r['n_events_evaluable']})" for r in t))


if __name__ == "__main__":
    main()
