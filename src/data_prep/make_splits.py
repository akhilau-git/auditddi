"""Generate every audited split and its frozen event vocabulary.

    python -m src.data_prep.make_splits \
        --table /content/drive/MyDrive/auditddi-data/event_table_v1 \
        --out   /content/drive/MyDrive/auditddi-data/splits_v1 \
        --seeds 0 1 2 3 4 --kinds transductive cold_drug scaffold random_pair

For each (kind, seed) it writes
    <kind>_seed<k>.npz / .json     partitions, sizes, audit checks, digest
    <kind>_seed<k>_vocab.csv       vocabulary frozen from TRAIN pairs only
    <kind>_seed<k>_support.csv     positives per partition and per event
and a top-level splits_manifest.json.  Any failed audit aborts the run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .build_event_table import freeze_vocabulary, load_event_table, pos_weights, vocabulary_hash
from .multilabel_splits import (
    assert_clean,
    cold_drug_split,
    event_support,
    random_pair_split,
    save_split,
    scaffold_split,
    scaled_min_pairs,
    transductive_split,
)

MIN_PAIRS_FULL = 750   # Guide 1 "~863 events" rule on the full 63,473-pair table


def make_one(kind: str, table, seed: int, frac_val: float, frac_test: float):
    pairs, drugs = table.pairs, table.drugs
    if kind == "random_pair":
        return random_pair_split(pairs, frac_val, frac_test, seed)
    if kind == "transductive":
        return transductive_split(pairs, frac_val, frac_test, seed)
    if kind == "cold_drug":
        return cold_drug_split(pairs, len(drugs), frac_val, frac_test, seed)
    if kind == "scaffold":
        return scaffold_split(pairs, drugs, frac_val, frac_test, seed)
    raise ValueError(kind)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--kinds", nargs="+", default=["transductive", "cold_drug", "scaffold", "random_pair"])
    ap.add_argument("--frac-val", type=float, default=0.10, help="pair fraction (pair splits) or drug fraction (cold/scaffold)")
    ap.add_argument("--frac-test", type=float, default=0.20)
    ap.add_argument("--min-pairs-full", type=int, default=MIN_PAIRS_FULL)
    a = ap.parse_args(list(argv) if argv is not None else None)

    table = load_event_table(a.table)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"table": str(a.table), "min_pairs_full": a.min_pairs_full, "splits": []}

    for kind in a.kinds:
        for seed in a.seeds:
            res = make_one(kind, table, seed, a.frac_val, a.frac_test)
            audit = assert_clean(res, table.pairs)          # raises on any leakage
            name = f"{kind}_seed{seed}"

            min_tr = scaled_min_pairs(a.min_pairs_full, len(res.train), table.n_pairs)
            vocab = freeze_vocabulary(table.Y, res.train, table.events, min_train_pairs=min_tr)
            if len(vocab) == 0:
                raise ValueError(
                    f"{kind} seed {seed}: no event has >= {min_tr} training pairs "
                    f"(--min-pairs-full {a.min_pairs_full} scaled to {len(res.train)}/{table.n_pairs} training pairs). "
                    "Lower --min-pairs-full or check the event table.")
            w = pos_weights(vocab)
            vocab["pos_weight"] = w
            vocab.to_csv(a.out / f"{name}_vocab.csv", index=False)

            parts = {"train": res.train}
            parts.update({f"val_{k}": v for k, v in res.val.items() if k != "all"} if "s2" in res.val else {"val": res.val["all"]})
            parts.update({f"test_{k}": v for k, v in res.test.items()})
            support = event_support(table.Y, vocab, parts)
            support.to_csv(a.out / f"{name}_support.csv", index=False)

            info = save_split(res, a.out, name, audit)
            info["vocab"] = {"n_events": int(len(vocab)), "min_train_pairs": int(min_tr), "sha256": vocabulary_hash(vocab)}
            info["evaluable_events"] = {
                c.replace("evaluable_", ""): int(support[c].sum()) for c in support.columns if c.startswith("evaluable_")
            }
            (a.out / f"{name}.json").write_text(json.dumps(info, indent=2))
            manifest["splits"].append({k: info[k] for k in ("name", "kind", "seed", "sizes", "sha256_partition_digest", "vocab", "evaluable_events")})
            print(f"{name:24s} train={info['sizes']['train']:6d} events={len(vocab):4d} audit=OK")

    (a.out / "splits_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
