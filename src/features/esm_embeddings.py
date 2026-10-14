"""ESM-2 protein embeddings and per-drug aggregation (Guide 1, section 4.4).

    python -m src.features.esm_embeddings \
        --sequences /content/drive/MyDrive/auditddi-data/uniprot_v2/target_sequences.fasta \
        --targets   /content/drive/MyDrive/auditddi-data/p1_targets/drug_targets_union.csv \
        --n-drugs 645 --out /content/drive/MyDrive/auditddi-data/esm_v1 --model facebook/esm2_t6_8M_UR50D

Run on a GPU runtime.  Per protein: mean of the residue embeddings (special tokens excluded).  Sequences longer
than --max-len (default 1022, ESM-2's limit) are TRUNCATED to their first --max-len residues and flagged.
Per drug: weighted mean over the drug's target embeddings; mechanism-of-action targets get weight --moa-weight,
other targets weight 1.  A drug without a target sequence gets a zero vector and has_esm = False; nothing is
imputed.  The embedder is injectable, so the aggregation logic is tested without a GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

Embedder = Callable[[List[str]], np.ndarray]


def read_fasta_simple(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    acc, buf = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if acc is not None:
                seqs[acc] = "".join(buf)
            acc, buf = line[1:].split()[0].split("|")[1] if "|" in line else line[1:].split()[0], []
        elif acc is not None:
            buf.append(line.strip())
    if acc is not None:
        seqs[acc] = "".join(buf)
    return seqs


def hf_embedder(model_name: str = "facebook/esm2_t6_8M_UR50D", device: Optional[str] = None, batch_size: int = 8, max_len: int = 1022) -> Embedder:
    """Mean-pooled ESM-2 residue embeddings via HuggingFace transformers (needs torch + transformers)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(dev).eval()

    @torch.no_grad()
    def run(seqs: List[str]) -> np.ndarray:
        out = []
        order = np.argsort([len(s) for s in seqs])                      # length-sorted batches waste less padding
        for i in range(0, len(seqs), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([seqs[j][:max_len] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len + 2).to(dev)
            h = model(**enc).last_hidden_state                          # [b, L, d]
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0                                              # <cls>
            lengths = enc["attention_mask"].sum(1)
            mask[torch.arange(len(idx)), lengths - 1] = 0               # <eos>
            m = mask.unsqueeze(-1).to(h.dtype)
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy())
        emb = np.concatenate(out)
        res = np.empty_like(emb)
        res[order] = emb
        return res

    return run


def embed_proteins(seqs: Dict[str, str], embedder: Embedder, model_tag: str, cache: Optional[Path] = None, max_len: int = 1022):
    """Returns ({accession: vector}, report).  Caches by (model_tag, sequence hash)."""
    cache_store: Dict[str, np.ndarray] = {}
    if cache is not None and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        cache_store = {k: z[k] for k in z.files}
    keys = {a: f"{model_tag}|{hashlib.sha256(s[:max_len].encode()).hexdigest()}" for a, s in seqs.items()}
    todo = [a for a in seqs if keys[a].replace("|", "_") not in cache_store and len(seqs[a]) > 0]
    if todo:
        vecs = embedder([seqs[a][:max_len] for a in todo])
        for a, v in zip(todo, vecs):
            cache_store[keys[a].replace("|", "_")] = np.asarray(v, dtype=np.float32)
        if cache is not None:
            np.savez_compressed(cache, **cache_store)
    emb = {a: cache_store[keys[a].replace("|", "_")] for a in seqs if keys[a].replace("|", "_") in cache_store}
    rep = {"proteins": len(seqs), "embedded_now": len(todo), "from_cache": len(emb) - len(todo), "truncated": int(sum(len(s) > max_len for s in seqs.values())),
           "empty_sequences": int(sum(len(s) == 0 for s in seqs.values()))}
    return emb, rep


def drug_embeddings(n_drugs: int, targets: pd.DataFrame, emb: Dict[str, np.ndarray], moa_weight: float = 2.0):
    """Weighted mean of a drug's target embeddings.  Returns (matrix [n_drugs, d], has_esm bool[n_drugs])."""
    d = len(next(iter(emb.values()))) if emb else 0
    M = np.zeros((n_drugs, d), dtype=np.float32)
    has = np.zeros(n_drugs, dtype=bool)
    t = targets.dropna(subset=["drug_idx", "uniprot"])
    has_moa = "moa" in t
    for drug, g in t.groupby("drug_idx"):
        vs, ws = [], []
        for r in g.itertuples():
            v = emb.get(str(r.uniprot))
            if v is None:
                continue
            moa = getattr(r, "moa", None) if has_moa else None
            vs.append(v)
            ws.append(moa_weight if str(moa) in ("1", "1.0", "True", "true") else 1.0)
        if vs:
            w = np.asarray(ws, dtype=np.float32)
            M[int(drug)] = (np.stack(vs) * w[:, None]).sum(0) / w.sum()
            has[int(drug)] = True
    return M, has


def main(argv: Optional[Sequence[str]] = None, embedder: Optional[Embedder] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", required=True, type=Path)
    ap.add_argument("--targets", required=True, type=Path)
    ap.add_argument("--n-drugs", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--moa-weight", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args(list(argv) if argv is not None else None)
    a.out.mkdir(parents=True, exist_ok=True)
    seqs = read_fasta_simple(a.sequences)
    emb_fn = embedder or hf_embedder(a.model, batch_size=a.batch_size, max_len=a.max_len)
    emb, rep = embed_proteins(seqs, emb_fn, a.model, a.out / "protein_embeddings_cache.npz", a.max_len)
    M, has = drug_embeddings(a.n_drugs, pd.read_csv(a.targets), emb, a.moa_weight)
    np.savez_compressed(a.out / "drug_esm.npz", esm=M, has_esm=has, model=np.array(a.model))
    rep.update({"drugs_with_esm": int(has.sum()), "dim": int(M.shape[1]), "model": a.model, "moa_weight": a.moa_weight})
    (a.out / "esm_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
