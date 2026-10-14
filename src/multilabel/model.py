"""AuditDDI multi-label model (Guide 1, sections 4.3-4.8).

        drug A features --+                                     +--> [E_A + E_B || |E_A - E_B| || E_A * E_B]
                          +--> per-drug encoder (shared) --> E --+                    |
        drug B features --+                                                           v
                                                         MLP head: Linear-BN-LeakyReLU-Dropout x2 -> n_events logits

Per-drug encoder (weights shared between A and B):
  * one small MLP tower per feature block (fingerprint, target, gene, target class, ATC, CYP, ESM-2 ...)
  * every BIOLOGY tower output is multiplied by that block's known/missing flag, so an unknown block contributes
    exactly nothing, whatever numbers sit in its feature columns (a test enforces this)
  * the masked biology outputs are fused by one linear layer; E = [fingerprint towers || fused biology]

The pair vector is built only from symmetric operations, so f(A, B) == f(B, A) exactly (a test enforces this).
No patient data enters the model.  Outputs are RAW LOGITS, one independent sigmoid per event; calibration is a
separate post-hoc step (src.trust).

Run the tests on a machine with torch:   python -m unittest tests.test_multilabel_model -v
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def _tower(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))


class AuditDDIML(nn.Module):
    def __init__(self, layout: Dict[str, dict], n_events: int, emb: int = 64, tower_hidden: int = 128, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.layout = {k: {"start": int(v["start"]), "end": int(v["end"]), "flag": None if v["flag"] is None else int(v["flag"])} for k, v in layout.items()}
        self.n_events, self.emb = int(n_events), int(emb)
        self.chem_names = [k for k, v in self.layout.items() if v["flag"] is None]
        self.bio_names = [k for k, v in self.layout.items() if v["flag"] is not None]
        self.chem = nn.ModuleDict({k: _tower(self.layout[k]["end"] - self.layout[k]["start"], tower_hidden, emb, dropout) for k in self.chem_names})
        self.bio = nn.ModuleDict({k: _tower(self.layout[k]["end"] - self.layout[k]["start"], tower_hidden, emb, dropout) for k in self.bio_names})
        self.bio_fuse = nn.Linear(emb * len(self.bio_names), emb) if self.bio_names else None
        self.drug_dim = emb * len(self.chem_names) + (emb if self.bio_names else 0)
        if self.drug_dim == 0:
            raise ValueError("the layout has no feature blocks")
        pair_dim = 3 * self.drug_dim
        self.head = nn.Sequential(
            nn.Linear(pair_dim, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(hidden, self.n_events),
        )

    # ---- per-drug encoder ------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        for k in self.chem_names:
            L = self.layout[k]
            parts.append(self.chem[k](x[:, L["start"]:L["end"]]))
        if self.bio_names:
            outs = []
            for k in self.bio_names:
                L = self.layout[k]
                h = self.bio[k](x[:, L["start"]:L["end"]])
                outs.append(h * x[:, L["flag"]:L["flag"] + 1])             # unknown block (flag 0) -> exactly zero
            parts.append(torch.relu(self.bio_fuse(torch.cat(outs, dim=1))))
        return torch.cat(parts, dim=1)

    @staticmethod
    def combine(ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant pair vector: sum, absolute difference, element-wise product."""
        return torch.cat([ea + eb, (ea - eb).abs(), ea * eb], dim=1)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        return self.head(self.combine(self.encode(xa), self.encode(xb)))

    def init_output_bias(self, prevalence) -> None:
        """Start every event at its training prevalence (logit of the base rate)."""
        p = torch.as_tensor(np.asarray(prevalence, dtype=np.float32)).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.log(p / (1 - p)).to(self.head[-1].bias.device))


def drop_biology(x: torch.Tensor, layout: Dict[str, dict], p: float) -> torch.Tensor:
    """Training-time augmentation: with probability p per drug row, mark EVERY biology block unknown
    (features and flags zeroed).  It teaches the model to cope with drugs it knows nothing about."""
    if p <= 0:
        return x
    keep = (torch.rand(x.shape[0], 1, device=x.device) >= p).to(x.dtype)
    x = x.clone()
    for L in layout.values():
        if L["flag"] is None:
            continue
        x[:, L["start"]:L["flag"] + 1] = x[:, L["start"]:L["flag"] + 1] * keep
    return x


@torch.no_grad()
def predict_proba(model: AuditDDIML, F: torch.Tensor, a_idx: torch.Tensor, b_idx: torch.Tensor, batch: int = 2048) -> np.ndarray:
    """Event probabilities for pairs (a_idx[i], b_idx[i]); F is the [n_drugs, d] feature matrix on the model's device."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(a_idx), batch):
        out.append(torch.sigmoid(model(F[a_idx[i:i + batch]], F[b_idx[i:i + batch]])).float().cpu())
    model.train(was_training)
    return torch.cat(out).numpy() if out else np.zeros((0, model.n_events), dtype=np.float32)


def save_checkpoint(path: Path, model: AuditDDIML, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"emb": model.emb, "n_events": model.n_events, "layout": model.layout,
           "tower_hidden": model.chem[next(iter(model.chem))][0].out_features if model.chem_names else model.bio[next(iter(model.bio))][0].out_features,
           "hidden": model.head[0].out_features, "dropout": float(model.head[3].p)}
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": cfg, "meta": meta}, path)
    path.with_suffix(".json").write_text(json.dumps({"config": cfg, "meta": meta}, indent=2, default=str))


def load_checkpoint(path: Path, device: Optional[str] = None):
    ck = torch.load(Path(path), map_location=device or "cpu")
    c = ck["config"]
    model = AuditDDIML(c["layout"], c["n_events"], c["emb"], c["tower_hidden"], c["hidden"], c["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model.to(device or "cpu"), ck["meta"]
