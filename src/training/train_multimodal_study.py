"""Unified Multimodal Study Suite for AuditDDI.

Includes:
1. Extended Training with Validation Checkpointing and Convergence Curve Tracking.
2. Modality Ablation Studies (Molecular Only vs +Genes vs +FAERS vs Full Multimodal).
3. S1 Cold-Start Error Analysis stratified by PharmGKB/FAERS Coverage Tiers.
4. Model Calibration Analysis (ECE, Brier score, Platt scaling).
5. Production Checkpoint Export.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import importlib
import hashlib
import random
import src.data_prep.cached_graph_loader as _cgl_mod
try:
    importlib.reload(_cgl_mod)
except Exception:
    pass

from src.data_prep.cached_graph_loader import (
    MolecularCache,
    build_cached_multimodal_dataloader,
)


def _make_dataloader(
    df: pd.DataFrame,
    cache: MolecularCache,
    batch_size: int = 128,
    shuffle: bool = True,
    neighbor_memory: Any = None,
) -> Any:
    """Safely build dataloader across dynamic module reloads."""
    import inspect
    sig = inspect.signature(build_cached_multimodal_dataloader)
    if 'neighbor_memory' in sig.parameters:
        return build_cached_multimodal_dataloader(df, cache, batch_size=batch_size, shuffle=shuffle, neighbor_memory=neighbor_memory)
    return build_cached_multimodal_dataloader(df, cache, batch_size=batch_size, shuffle=shuffle)
from src.models.calibration import (
    apply_calibrator,
    expected_calibration_error,
    fit_platt_calibrator,
)
from src.models.ddi_model import (
    MODEL_ARCHITECTURE_ABLATION_GENES,
    MODEL_ARCHITECTURE_EDGE_AWARE,
    MODEL_ARCHITECTURE_MULTIMODAL,
    AuditDDIModel,
    PxDDIModel,
)
from src.training.benchmark_cold_start import (
    ensure_benchmark_splits,
    evaluate_loader,
    safe_forward_multimodal,
)


def predict_loader(
    model: AuditDDIModel,
    loader: Any,
    device: torch.device,
    is_multimodal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw probabilities and ground truth targets from a DataLoader."""
    model.eval()
    all_scores: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for batch in loader:
            da = batch['drug_a'].to(device)
            db = batch['drug_b'].to(device)
            lbls = batch['labels'].cpu().numpy().ravel()

            if is_multimodal:
                risk_logits, _, _ = safe_forward_multimodal(model, batch, da, db, device)
            else:
                risk_logits, _, _ = model(drug_a=da, drug_b=db)

            probs = torch.sigmoid(risk_logits).cpu().numpy().ravel()
            all_scores.extend(probs.tolist())
            all_targets.extend(lbls.tolist())

    return np.array(all_scores, dtype=float), np.array(all_targets, dtype=float)


def evaluate_predictions(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float | str = 'optimal',
) -> dict[str, float]:
    """Compute comprehensive performance metrics with optimal or specified threshold."""
    if len(np.unique(targets)) < 2:
        return {
            'auroc': 0.5,
            'auprc': float(np.mean(targets)) if len(targets) else 0.0,
            'f1': 0.0,
            'accuracy': 0.5,
            'mcc': 0.0,
            'brier': 0.25,
            'optimal_threshold': 0.5,
            'false_negatives': 0,
            'false_positives': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'fnr': 0.0,
            'fpr': 0.0,
        }

    auroc = float(roc_auc_score(targets, scores))
    auprc = float(average_precision_score(targets, scores))
    brier = float(brier_score_loss(targets, scores))

    if threshold == 'optimal' or threshold is None:
        try:
            fpr_arr, tpr_arr, thresh_arr = roc_curve(targets, scores)
            # Cost-sensitive Youden index matching pos_weight=2.0 (penalty: -2 FN / -1 FP)
            # Prioritizes clinical sensitivity/recall (>70% standard) over false positive alarms
            j_scores = 2.0 * tpr_arr - fpr_arr
            best_idx = int(np.argmax(j_scores)) if len(j_scores) else 0
            opt_thresh = float(thresh_arr[best_idx]) if len(thresh_arr) > best_idx else 0.38
            opt_thresh = max(min(opt_thresh, 0.45), 0.20)
        except Exception:
            opt_thresh = 0.38
    else:
        opt_thresh = float(threshold)

    preds = (scores >= opt_thresh).astype(int)
    acc = float(accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, zero_division=0))
    mcc = float(matthews_corrcoef(targets, preds))

    pos_mask = (targets == 1)
    neg_mask = (targets == 0)
    fn = int(np.sum((preds == 0) & pos_mask))
    fp = int(np.sum((preds == 1) & neg_mask))
    tp = int(np.sum((preds == 1) & pos_mask))
    tn = int(np.sum((preds == 0) & neg_mask))
    fnr = float(fn / max(pos_mask.sum(), 1))
    fpr = float(fp / max(neg_mask.sum(), 1))
    recall = float(tp / max(pos_mask.sum(), 1))
    sensitivity = recall
    specificity = float(tn / max(neg_mask.sum(), 1))

    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'f1': f1,
        'mcc': mcc,
        'brier': brier,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'optimal_threshold': opt_thresh,
        'false_negatives': fn,
        'false_positives': fp,
        'true_positives': tp,
        'true_negatives': tn,
        'fnr': fnr,
        'fpr': fpr,
    }


def train_extended_multimodal(
    cache: MolecularCache,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    cold_dev_splits: dict[str, pd.DataFrame] | None = None,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    architecture_version: str = MODEL_ARCHITECTURE_MULTIMODAL,
    device: torch.device | None = None,
    chembl_pretrained_path: str | Path | None = None,
    use_cross_modal_attention: bool = True,
    use_cross_drug_attention: bool = False,
    use_target_encoder: bool = True,
    use_faers_features: bool = False,
    use_neighbor_memory: bool = False,
    select_best_by: str = 'val',
    pos_weight: float = 2.2,
    use_ssl: bool = False,
    ssl_weight: float = 0.2,
    ssl_pairs_count: int = 5000,
    memory_dropout: float = 0.75,
    embedding_noise_std: float = 0.02,
    patience: int = 8,
    hidden_dim: int = 64,
    **kwargs: Any,
) -> tuple[AuditDDIModel, pd.DataFrame, dict[str, Any]]:
    """Train the multimodal model across extended epochs with checkpointing."""
    if chembl_pretrained_path is not None:
        raise ValueError(
            'This multimodal benchmark uses a different split builder than the audited '
            'ChEMBL pretraining contract. Do not load that checkpoint here; evaluate '
            'ChEMBL with src/training/run_experiment_suite.py, which validates split provenance.'
        )
    if use_faers_features:
        raise ValueError(
            'FAERS inputs are disabled for DDI prediction because the current toxicity '
            'bridge has no verified as-of date. Add a time-filtered FAERS bridge and '
            'overlap audit before enabling this feature.'
        )
    if select_best_by == 's1':
        print("⚠️ Audit Notice: 'select_best_by=s1' selects models using S1 test split (test data leakage). "
              "AuditDDI policy requires model selection and early stopping on validation/dev split. "
              "Automatically switching to select_best_by='val'.")
        select_best_by = 'val'
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    is_multimodal = (architecture_version != MODEL_ARCHITECTURE_EDGE_AWARE)

    neighbor_mem = None
    if use_neighbor_memory:
        from src.models.neighbor_memory import AuditableNeighborMemory
        src_col = 'drug_a_id' if 'drug_a_id' in train_df.columns else ('source' if 'source' in train_df.columns else train_df.columns[0])
        tgt_col = 'drug_b_id' if 'drug_b_id' in train_df.columns else ('target' if 'target' in train_df.columns else train_df.columns[1])
        lbl_col = 'label' if 'label' in train_df.columns else train_df.columns[-1]
        neighbor_mem = AuditableNeighborMemory(k_neighbors=5)
        neighbor_mem.fit(train_df[src_col].tolist(), train_df[tgt_col].tolist(), train_df[lbl_col].tolist())
        print(f"AuditableNeighborMemory fitted on {len(train_df)} training edges with {len(neighbor_mem.training_smiles)} unique drugs.")

    train_loader = _make_dataloader(train_df, cache, batch_size=batch_size, shuffle=True, neighbor_memory=neighbor_mem)
    val_loader = _make_dataloader(val_df, cache, batch_size=batch_size, shuffle=False, neighbor_memory=neighbor_mem)

    test_loaders = {
        name: _make_dataloader(df, cache, batch_size=batch_size, shuffle=False, neighbor_memory=neighbor_mem)
        for name, df in test_splits.items()
    }

    # Semi-Supervised Learning (SSL) on unobserved non-test drug pairs
    ssl_loader = None
    source_col = 'drug_a_id' if 'drug_a_id' in train_df.columns else ('source' if 'source' in train_df.columns else train_df.columns[0])
    target_col = 'drug_b_id' if 'drug_b_id' in train_df.columns else ('target' if 'target' in train_df.columns else train_df.columns[1])
    training_drugs = set(train_df[source_col].astype(str)) | set(train_df[target_col].astype(str))
    if use_ssl and is_multimodal:
        try:
            known_pairs: set[tuple[str, str]] = set()
            # Pseudo-label pair generation is restricted to training identities;
            # validation/test rows and held-out drug identities are never read.
            for _, r in train_df.iterrows():
                sa, sb = str(r[source_col]).strip(), str(r[target_col]).strip()
                known_pairs.add((sa, sb))
                known_pairs.add((sb, sa))

            valid_drugs = [s for s in training_drugs if s in cache.graphs and s in cache.fingerprints]
            if len(valid_drugs) >= 10:
                rng = np.random.RandomState(42)
                ssl_pairs: list[dict[str, Any]] = []
                attempts = 0
                max_attempts = ssl_pairs_count * 10
                while len(ssl_pairs) < ssl_pairs_count and attempts < max_attempts:
                    attempts += 1
                    i, j = rng.choice(len(valid_drugs), size=2, replace=False)
                    d1, d2 = valid_drugs[i], valid_drugs[j]
                    if (d1, d2) not in known_pairs:
                        ssl_pairs.append({'drug_a_id': d1, 'drug_b_id': d2, 'label': 0.0})
                        known_pairs.add((d1, d2))
                        known_pairs.add((d2, d1))

                if len(ssl_pairs) >= 50:
                    ssl_df = pd.DataFrame(ssl_pairs)
                    ssl_loader = _make_dataloader(ssl_df, cache, batch_size=batch_size, shuffle=True, neighbor_memory=neighbor_mem)
                    print(f"Semi-Supervised Learning (SSL) initialized with {len(ssl_df)} unobserved non-test pairs.")
        except Exception as ssl_err:
            print(f"SSL initialization notice: {ssl_err}")

    sample_batch = next(iter(train_loader))
    in_channels = sample_batch['drug_a'].x.size(1)
    edge_dim = sample_batch['drug_a'].edge_attr.size(1)

    hidden_dim = int(kwargs.get('hidden_dim', 64))
    model = AuditDDIModel(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        edge_feature_dim=edge_dim,
        architecture_version=architecture_version,
        gene_feature_dim=cache.gene_dim,
        gene_hidden_channels=64,
        use_clinical_toxicity=is_multimodal and use_faers_features,
        use_cross_modal_attention=use_cross_modal_attention if is_multimodal else False,
        use_cross_modal_target_attention=is_multimodal and use_target_encoder,
        use_cross_modal_pdb_attention=is_multimodal,
        use_inductive_bio_features=is_multimodal,
        use_fusion_norm=is_multimodal,
        use_cross_drug_attention=use_cross_drug_attention,
        use_target_encoder=use_target_encoder,
        target_feature_dim=cache.target_dim,
        target_hidden_channels=64,
        use_pdb_encoder=is_multimodal,
        pdb_feature_dim=cache.pdb_dim,
        pdb_hidden_channels=64,
        use_neighbor_memory=use_neighbor_memory,
        use_geo_features=is_multimodal,
        use_geo_encoder=is_multimodal,
        geo_dim=cache.geo_dim,
        geo_hidden_channels=32,
        memory_dropout=memory_dropout,
        embedding_noise_std=embedding_noise_std,
        use_protein_sequence_encoder=is_multimodal and bool(kwargs.get('use_protein_sequence_encoder', False)),
    )

    encoder_warmed = False
    if not encoder_warmed and hasattr(model, 'encoder'):
        from src.models.encoder import EdgeAwareMolecularEncoder
        from src.models.encoder_pretraining import (
            EdgeAwareContrastivePretrainer,
            augment_edge_aware_batch,
            bidirectional_nt_xent_loss,
        )
        if isinstance(model.encoder, EdgeAwareMolecularEncoder) and len(cache.graphs) >= 4:
            print("🚀 Running self-supervised molecular graph contrastive warm-up (EdgeAware NT-Xent on cached graphs, 15 epochs)...")
            try:
                from torch_geometric.data import Batch
                pretrainer = EdgeAwareContrastivePretrainer(
                    in_channels=in_channels,
                    edge_feature_dim=edge_dim,
                    hidden_channels=hidden_dim,
                ).to(device)
                pt_opt = AdamW(pretrainer.parameters(), lr=1e-3, weight_decay=1e-4)
                # Never warm the encoder on held-out validation/test molecules.
                graph_list = [cache.graphs[s] for s in training_drugs if s in cache.graphs]
                pretrainer.train()
                for _ in range(15):
                    perm = torch.randperm(len(graph_list)).tolist()
                    pt_batch_sz = min(64, len(graph_list))
                    for i in range(0, len(graph_list), pt_batch_sz):
                        sub_graphs = [graph_list[idx] for idx in perm[i:i + pt_batch_sz]]
                        if len(sub_graphs) < 2:
                            continue
                        pyg_batch = cast(Any, Batch.from_data_list(sub_graphs)).to(device)
                        v1 = augment_edge_aware_batch(pyg_batch, atom_feature_mask_rate=0.15, bond_feature_mask_rate=0.15)
                        v2 = augment_edge_aware_batch(pyg_batch, atom_feature_mask_rate=0.15, bond_feature_mask_rate=0.15)
                        z1 = pretrainer(v1)
                        z2 = pretrainer(v2)
                        pt_loss = bidirectional_nt_xent_loss(z1, z2, temperature=0.2)
                        pt_opt.zero_grad()
                        pt_loss.backward()
                        pt_opt.step()
                model.encoder.load_state_dict(pretrainer.encoder.state_dict())
                encoder_warmed = True
                print("✅ Self-supervised molecular encoder warm-up completed successfully (learned graph representations).")
            except Exception as wu_err:
                print(f"Self-supervised warm-up notice ({wu_err}); proceeding with random initialization.")

    model = model.to(device)

    if encoder_warmed:
        encoder_params = list(model.encoder.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
        optimizer = AdamW([
            {'params': encoder_params, 'lr': learning_rate * 0.1},
            {'params': other_params, 'lr': learning_rate},
        ], weight_decay=weight_decay)
        print(f"Discriminative LR: encoder LR={learning_rate * 0.1:.1e}, fusion heads LR={learning_rate:.1e}")
    else:
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    pos_weight_tensor = torch.tensor([pos_weight], device=device) if pos_weight > 1.0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(f"Loss Function: BCEWithLogitsLoss (pos_weight={pos_weight:.1f}, reflecting -{pos_weight:.0f} FN / -1 FP asymmetric penalty)")

    history_records: list[dict[str, Any]] = []
    best_val_auroc = -1.0
    best_val_epoch = 0
    epochs_without_val_improvement = 0
    best_weights_path = out_p / f'{architecture_version}_best.pt'

    print(f"\n{'=' * 80}")
    print(f"STARTING EXTENDED TRAINING: {architecture_version} ({epochs} epochs on {device})")
    print(f"{'=' * 80}")

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        ssl_iter = iter(ssl_loader) if ssl_loader is not None else None

        for batch in train_loader:
            optimizer.zero_grad()
            da = batch['drug_a'].to(device)
            db = batch['drug_b'].to(device)
            labels = batch['labels'].to(device)

            if is_multimodal:
                risk_logits, _, _ = safe_forward_multimodal(model, batch, da, db, device)
            else:
                risk_logits, _, _ = model(drug_a=da, drug_b=db)

            # Asymmetric Focal loss modulation with gamma=2.0 and pos_weight to strongly penalize false negatives
            smoothed_labels = labels * 0.96 + 0.02
            targets = smoothed_labels.view(-1)
            logits_flat = risk_logits.view(-1)
            probs = torch.sigmoid(logits_flat)
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            focal_mod = torch.pow((1.0 - p_t).clamp(min=0.05, max=1.0), 2.0)
            raw_bce = F.binary_cross_entropy_with_logits(
                logits_flat, targets, pos_weight=pos_weight_tensor, reduction='none'
            )
            supervised_loss = (focal_mod * raw_bce).mean()
            total_batch_loss = supervised_loss

            # Multi-Dataset Biological Contrastive Alignment Loss (PharmGKB, BindingDB, GEO, FAERS, PubChem, PDB)
            if is_multimodal and hasattr(model, 'encoder'):
                try:
                    batch_size_cur = da.num_graphs if hasattr(da, 'num_graphs') else da.x.size(0)
                    bio_sims: list[tuple[torch.Tensor, torch.Tensor, float]] = []

                    # 1. PharmGKB CYP Enzymes & Transporters (Primary Pharmacogenomic DDI Driver)
                    if 'gene_a' in batch and 'gene_b' in batch:
                        ga = batch['gene_a'].to(device)
                        gb = batch['gene_b'].to(device)
                        gmask = (batch['gene_mask_a'].to(device) > 0.5) & (batch['gene_mask_b'].to(device) > 0.5)
                        if gmask.any():
                            gene_sim = F.cosine_similarity(ga, gb, dim=-1).clamp(0.0, 1.0)
                            bio_sims.append((gene_sim, gmask, 1.5))

                    # 2. BindingDB Target Affinity Vectors (Receptor / Kinase Competition)
                    if 'target_a' in batch and 'target_b' in batch:
                        ta = batch['target_a'].to(device)
                        tb = batch['target_b'].to(device)
                        tmask = (batch['target_mask_a'].to(device) > 0.5) & (batch['target_mask_b'].to(device) > 0.5)
                        if tmask.any():
                            target_sim = F.cosine_similarity(ta, tb, dim=-1).clamp(0.0, 1.0)
                            bio_sims.append((target_sim, tmask, 1.2))

                    # 3. GEO Disease Transcriptomics
                    if 'geo_a' in batch and 'geo_b' in batch:
                        geoa = batch['geo_a'].to(device)
                        geob = batch['geo_b'].to(device)
                        geomask = (batch['geo_mask_a'].to(device) > 0.5) & (batch['geo_mask_b'].to(device) > 0.5)
                        if geomask.any():
                            geo_sim = F.cosine_similarity(geoa, geob, dim=-1).clamp(0.0, 1.0)
                            bio_sims.append((geo_sim, geomask, 0.2))

                    # 4. FAERS Clinical Adverse Event Proximity
                    if use_faers_features and 'tox_a' in batch and 'tox_b' in batch:
                        toxa = batch['tox_a'].to(device).float()
                        toxb = batch['tox_b'].to(device).float()
                        toxmask = (batch['tox_mask_a'].to(device) > 0.5) & (batch['tox_mask_b'].to(device) > 0.5)
                        if toxmask.any():
                            tox_sim = (1.0 - torch.abs(toxa - toxb).clamp(0.0, 1.0))
                            bio_sims.append((tox_sim, toxmask, 0.2))

                    # 5. PubChem ECFP Morgan Structural Proximity
                    if 'fp_a' in batch and 'fp_b' in batch:
                        fpa = batch['fp_a'].to(device).float()
                        fpb = batch['fp_b'].to(device).float()
                        fp_sim = F.cosine_similarity(fpa, fpb, dim=-1).clamp(0.0, 1.0)
                        fp_mask = torch.ones(batch_size_cur, dtype=torch.bool, device=device)
                        bio_sims.append((fp_sim, fp_mask, 0.5))

                    # 6. PDB 3D Macromolecular Co-Crystal & Target Proximity
                    if 'pdb_a' in batch and 'pdb_b' in batch:
                        pdba = batch['pdb_a'].to(device).float()
                        pdbb = batch['pdb_b'].to(device).float()
                        pdbmask = (batch['pdb_mask_a'].to(device) > 0.5) & (batch['pdb_mask_b'].to(device) > 0.5)
                        if pdbmask.any():
                            pdb_sim = F.cosine_similarity(pdba, pdbb, dim=-1).clamp(0.0, 1.0)
                            bio_sims.append((pdb_sim, pdbmask, 1.25))

                    if bio_sims:
                        batch_size_cur = da.num_graphs if hasattr(da, 'num_graphs') else da.x.size(0)
                        composite_bio = torch.zeros(batch_size_cur, device=device)
                        total_weight = torch.zeros(batch_size_cur, device=device)

                        for sim_vec, mask_vec, w in bio_sims:
                            m_flt = mask_vec.float()
                            composite_bio = composite_bio + sim_vec * m_flt * w
                            total_weight = total_weight + m_flt * w

                        valid_pairs = total_weight > 0
                        if valid_pairs.sum() > 1:
                            target_bio_sim = composite_bio[valid_pairs] / total_weight[valid_pairs].clamp(min=1e-5)
                            if hasattr(model, '_last_ea') and model._last_ea is not None and model._last_eb is not None:
                                ma = model._last_ea[valid_pairs]
                                mb = model._last_eb[valid_pairs]
                            else:
                                ma = model.encoder(da.x, da.edge_index, da.edge_attr, da.batch)[valid_pairs]
                                mb = model.encoder(db.x, db.edge_index, db.edge_attr, db.batch)[valid_pairs]
                            mol_sim = F.cosine_similarity(ma, mb, dim=-1).clamp(0.0, 1.0)
                            bio_loss = F.mse_loss(mol_sim, target_bio_sim)
                            bio_align_weight = getattr(model, 'bio_align_weight', 0.10)
                            total_batch_loss = total_batch_loss + bio_align_weight * bio_loss
                except Exception:
                    pass

            # Semi-Supervised Consistency Regularization
            if ssl_iter is not None and ssl_loader is not None:
                try:
                    ssl_batch = next(ssl_iter)
                except StopIteration:
                    ssl_iter = iter(ssl_loader)
                    ssl_batch = next(ssl_iter)

                ssl_da = ssl_batch['drug_a'].to(device)
                ssl_db = ssl_batch['drug_b'].to(device)
                if is_multimodal:
                    ssl_logits, _, _ = safe_forward_multimodal(model, ssl_batch, ssl_da, ssl_db, device)
                else:
                    ssl_logits, _, _ = model(drug_a=ssl_da, drug_b=ssl_db)

                ssl_probs = torch.sigmoid(ssl_logits.view(-1))
                high_conf_mask = (ssl_probs > 0.85) | (ssl_probs < 0.15)
                if high_conf_mask.sum() > 0:
                    pseudo_labels = (ssl_probs[high_conf_mask] > 0.50).float()
                    ssl_loss = criterion(ssl_logits.view(-1)[high_conf_mask], pseudo_labels)
                    total_batch_loss = supervised_loss + ssl_weight * ssl_loss

            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(total_batch_loss.item())

        scheduler.step()
        ep_sec = time.perf_counter() - ep_start
        avg_loss = total_loss / max(len(train_loader), 1)

        val_scores, val_targets = predict_loader(model, val_loader, device, is_multimodal=is_multimodal)
        val_metrics = evaluate_predictions(val_scores, val_targets, threshold='optimal')
        cold_dev_metrics: dict[str, dict[str, float]] = {}
        for split_name, dev_frame in (cold_dev_splits or {}).items():
            if dev_frame.empty:
                continue
            dev_loader = _make_dataloader(dev_frame, cache, batch_size=batch_size, shuffle=False, neighbor_memory=neighbor_mem)
            dev_scores, dev_targets = predict_loader(model, dev_loader, device, is_multimodal=is_multimodal)
            cold_dev_metrics[split_name] = evaluate_predictions(dev_scores, dev_targets, threshold='optimal')
        selection_values = [val_metrics['auroc']]
        selection_values.extend(
            cold_dev_metrics[name]['auroc']
            for name in ('s1_dev', 's2_dev')
            if name in cold_dev_metrics and len(np.unique(
                np.asarray(cold_dev_splits[name]['label'], dtype=float)
            )) > 1
        )
        selection_score = float(np.mean(selection_values))

        record = {
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_auroc': val_metrics['auroc'],
            'val_auprc': val_metrics['auprc'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'selection_score_macro_auroc': selection_score,
            **{
                f'{name}_{metric}': value
                for name, metrics in cold_dev_metrics.items()
                for metric, value in metrics.items()
            },
        }
        history_records.append(record)

        is_best = selection_score > best_val_auroc
        if is_best:
            best_val_auroc = selection_score
            best_val_epoch = epoch
            epochs_without_val_improvement = 0
            best_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': best_val_auroc,
                'model_selection_metric': 'macro_auroc_transductive_validation_s1_dev_s2_dev',
                'val_accuracy': val_metrics['accuracy'],
                'optimal_threshold': val_metrics.get('optimal_threshold', 0.35),
                'split_thresholds': {
                    'transductive': float(val_metrics.get('optimal_threshold', 0.5)),
                    **{
                        's1_cold' if name == 's1_dev' else 's2_semi': float(metrics['optimal_threshold'])
                        for name, metrics in cold_dev_metrics.items()
                        if name in {'s1_dev', 's2_dev'}
                    },
                },
                'in_channels': in_channels,
                'hidden_channels': hidden_dim,
                'edge_feature_dim': edge_dim,
                'architecture_version': architecture_version,
                'gene_feature_dim': cache.gene_dim,
                'gene_hidden_channels': 64,
                'use_clinical_toxicity': is_multimodal and use_faers_features,
                'use_neighbor_memory': use_neighbor_memory,
                'use_target_encoder': use_target_encoder,
                'target_feature_dim': cache.target_dim,
                'target_hidden_channels': 64,
                'use_pdb_encoder': is_multimodal,
                'pdb_feature_dim': cache.pdb_dim,
                'pdb_hidden_channels': 64,
                'use_geo_features': is_multimodal,
                'use_geo_encoder': is_multimodal,
                'geo_dim': cache.geo_dim,
                'geo_hidden_channels': 32,
                'use_cross_modal_attention': use_cross_modal_attention if is_multimodal else False,
                'use_cross_modal_target_attention': is_multimodal and use_target_encoder,
                'use_cross_modal_pdb_attention': is_multimodal,
                'use_inductive_bio_features': is_multimodal,
                'use_fusion_norm': is_multimodal,
                'use_cross_drug_attention': use_cross_drug_attention,
                'memory_dropout': memory_dropout,
                'embedding_noise_std': embedding_noise_std,
            }
            torch.save(best_dict, best_weights_path)
        else:
            epochs_without_val_improvement += 1

        best_mark = " [* Best Val]" if is_best else ""
        print(f"  Epoch {epoch:02d}/{epochs:02d} ({ep_sec:.1f}s) - Loss: {avg_loss:.4f} | "
              f"Val AUROC: {val_metrics['auroc']:.4f} (Acc: {val_metrics['accuracy']*100:.1f}%){best_mark}")

        # Early Stopping: Based strictly on validation split (zero test set leakage)
        if patience > 0 and epochs_without_val_improvement >= patience and epoch >= 4:
            print(f"\n⏹️ Early stopping triggered at Epoch {epoch}: Development selection score has not improved for {patience} consecutive epochs (Peak macro AUROC: {best_val_auroc:.4f} at Epoch {best_val_epoch}). Restoring best development checkpoint.")
            break

    # Load peak validation checkpoint for honest out-of-sample evaluation
    target_weights_path = best_weights_path
    if target_weights_path.is_file():
        ckpt = torch.load(target_weights_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\nLoaded best development checkpoint from epoch {ckpt['epoch']} (selection AUROC: {ckpt.get('val_auroc', 'N/A')})")

    history_df = pd.DataFrame(history_records)
    history_df.to_csv(out_p / f'{architecture_version}_training_history.csv', index=False)

    # Final split evaluation
    frozen_threshold = float(ckpt.get('optimal_threshold', 0.5)) if target_weights_path.is_file() else 0.5
    split_thresholds = ckpt.get('split_thresholds', {}) if target_weights_path.is_file() else {}
    final_results: dict[str, Any] = {
        'architecture': architecture_version,
        'best_val_auroc': best_val_auroc,
        'best_val_epoch': best_val_epoch,
        'validation_selected_threshold': frozen_threshold,
        'val_optimal_threshold': frozen_threshold,
        'test_evaluation_protocol': 'single_evaluation_after_macro_auroc_selection_on_validation_and_cold_dev; thresholds_frozen_from_matching_dev_splits',
    }
    for name, loader in test_loaders.items():
        scores, targets = predict_loader(model, loader, device, is_multimodal=is_multimodal)
        split_threshold = float(split_thresholds.get(name, frozen_threshold))
        m = evaluate_predictions(scores, targets, threshold=split_threshold)
        for k, v in m.items():
            final_results[f'{name}_{k}'] = v

    return model, history_df, final_results


def run_modality_ablation_study(
    cache: MolecularCache,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    cold_dev_splits: dict[str, pd.DataFrame] | None = None,
    epochs: int = 5,
    batch_size: int = 64,
    device: torch.device | None = None,
    chembl_pretrained_path: str | Path | None = None,
    use_neighbor_memory: bool = True,
    select_best_by: str = 'val',
    pos_weight: float = 1.75,
    use_ssl: bool = False,
    use_target_encoder: bool = True,
    use_faers_features: bool = False,
    memory_dropout: float = 0.75,
) -> pd.DataFrame:
    """Systematically run all 4 modality ablation variants and report deltas."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    ablation_variants = [
        ('Molecular Only (Baseline)', MODEL_ARCHITECTURE_EDGE_AWARE),
        ('Molecular + PharmGKB Genes', MODEL_ARCHITECTURE_ABLATION_GENES),
        ('Full Multimodal (AuditDDI)', MODEL_ARCHITECTURE_MULTIMODAL),
    ]

    all_ablation_results: list[dict[str, Any]] = []

    print(f"\n{'=' * 80}")
    print("STARTING SYSTEMATIC MODALITY ABLATION STUDY")
    print(f"{'=' * 80}")

    for display_name, arch in ablation_variants:
        print(f"\n--> Training Variant: {display_name} ({arch})...")
        use_tgt = use_target_encoder if arch == MODEL_ARCHITECTURE_MULTIMODAL else False
        _, _, results = train_extended_multimodal(
            cache=cache,
            train_df=train_df,
            val_df=val_df,
            test_splits=test_splits,
            cold_dev_splits=cold_dev_splits,
            output_dir=out_p / 'ablation_checkpoints',
            epochs=epochs,
            batch_size=batch_size,
            architecture_version=arch,
            device=device,
            chembl_pretrained_path=chembl_pretrained_path,
            use_neighbor_memory=use_neighbor_memory,
            select_best_by=select_best_by,
            pos_weight=pos_weight,
            use_ssl=use_ssl,
            use_target_encoder=use_tgt,
            use_faers_features=use_faers_features,
            memory_dropout=memory_dropout,
        )
        results['variant_name'] = display_name
        all_ablation_results.append(results)

    ablation_df = pd.DataFrame(all_ablation_results)

    # Compute deltas relative to baseline
    baseline_s1 = ablation_df.loc[ablation_df['architecture'] == MODEL_ARCHITECTURE_EDGE_AWARE, 's1_cold_auroc'].values[0]
    baseline_trans = ablation_df.loc[ablation_df['architecture'] == MODEL_ARCHITECTURE_EDGE_AWARE, 'transductive_auroc'].values[0]

    ablation_df['delta_s1_auroc'] = ablation_df['s1_cold_auroc'] - baseline_s1
    ablation_df['delta_transductive_auroc'] = ablation_df['transductive_auroc'] - baseline_trans

    csv_path = out_p / 'ablation_study_results.csv'
    ablation_df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 80}")
    print("ABLATION STUDY SUMMARY (QUANTIFIED MODALITY CONTRIBUTIONS):")
    print(f"{'=' * 80}")
    cols = ['variant_name', 's1_cold_auroc', 'delta_s1_auroc', 'transductive_auroc', 'delta_transductive_auroc']
    print(ablation_df[[c for c in cols if c in ablation_df.columns]].to_string(index=False))
    return ablation_df


def analyze_cold_start_coverage_errors(
    model: AuditDDIModel,
    cache: MolecularCache,
    s1_test_df: pd.DataFrame,
    output_dir: str | Path,
    device: torch.device | None = None,
    neighbor_memory: Any = None,
    optimal_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect misclassifications on unseen cold-start pairs stratified by external coverage."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if s1_test_df.empty:
        empty_annotated = pd.DataFrame(columns=pd.Index([
            'drug_a', 'drug_b', 'true_label', 'pred_prob', 'binary_pred',
            'is_correct', 'error_type', 'coverage_tier', 'gene_a', 'gene_b', 'faers_a', 'faers_b', 'geo_a', 'geo_b'
        ]))
        empty_summary = pd.DataFrame(columns=pd.Index([
            'coverage_tier', 'pair_count', 'accuracy', 'auroc', 'fpr', 'fnr', 'false_positives', 'false_negatives'
        ]))
        empty_annotated.to_csv(out_p / 'cold_start_error_analysis.csv', index=False)
        empty_summary.to_csv(out_p / 'cold_start_coverage_summary.csv', index=False)
        return empty_annotated, empty_summary

    loader = _make_dataloader(s1_test_df, cache, batch_size=64, shuffle=False, neighbor_memory=neighbor_memory)
    scores, targets = predict_loader(model, loader, device, is_multimodal=True)

    if optimal_threshold is None:
        try:
            from sklearn.metrics import roc_curve
            fpr_arr, tpr_arr, thresh_arr = roc_curve(targets, scores)
            j_scores = 2.0 * tpr_arr - fpr_arr
            best_idx = int(np.argmax(j_scores)) if len(j_scores) else 0
            optimal_threshold = float(thresh_arr[best_idx]) if len(thresh_arr) > best_idx else 0.38
            optimal_threshold = max(min(optimal_threshold, 0.45), 0.20)
        except Exception:
            optimal_threshold = 0.38

    preds = (scores >= optimal_threshold).astype(int)

    # Annotate coverage tier per pair
    # Resolve columns
    src_col = 'drug_a_id' if 'drug_a_id' in s1_test_df.columns else 'source'
    dst_col = 'drug_b_id' if 'drug_b_id' in s1_test_df.columns else 'target'

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(s1_test_df.itertuples(index=False)):
        sa = getattr(row, src_col)
        sb = getattr(row, dst_col)
        lbl = float(targets[idx])
        prob = float(scores[idx])
        pred = int(preds[idx])

        has_gene_a = cache.gene_masks.get(sa, torch.tensor(0.0)).item() > 0.5
        has_gene_b = cache.gene_masks.get(sb, torch.tensor(0.0)).item() > 0.5
        has_tox_a = cache.toxicity_masks.get(sa, torch.tensor(0.0)).item() > 0.5
        has_tox_b = cache.toxicity_masks.get(sb, torch.tensor(0.0)).item() > 0.5
        has_target_a = cache.target_masks.get(sa, torch.tensor(0.0)).item() > 0.5
        has_target_b = cache.target_masks.get(sb, torch.tensor(0.0)).item() > 0.5
        has_geo_a = cache.geo_masks.get(sa, torch.tensor(0.0)).item() > 0.5
        has_geo_b = cache.geo_masks.get(sb, torch.tensor(0.0)).item() > 0.5

        has_any_ext_a = has_gene_a or has_tox_a or has_target_a or has_geo_a
        has_any_ext_b = has_gene_b or has_tox_b or has_target_b or has_geo_b

        if has_any_ext_a and has_any_ext_b:
            tier = 'Both Drugs Profiled'
        elif has_any_ext_a or has_any_ext_b:
            tier = 'One Drug Profiled'
        else:
            tier = 'Zero External Coverage'

        err_type = 'Correct'
        if pred == 1 and lbl == 0:
            err_type = 'False Positive'
        elif pred == 0 and lbl == 1:
            err_type = 'False Negative'

        rows.append({
            'drug_a': sa,
            'drug_b': sb,
            'true_label': lbl,
            'pred_prob': prob,
            'binary_pred': pred,
            'is_correct': (pred == lbl),
            'error_type': err_type,
            'coverage_tier': tier,
            'gene_a': has_gene_a,
            'gene_b': has_gene_b,
            'faers_a': has_tox_a,
            'faers_b': has_tox_b,
            'bindingdb_a': has_target_a,
            'bindingdb_b': has_target_b,
            'geo_a': has_geo_a,
            'geo_b': has_geo_b,
        })

    annotated_df = pd.DataFrame(rows)
    annotated_df.to_csv(out_p / 'cold_start_error_analysis.csv', index=False)

    # Compute stratified summary metrics per coverage tier
    tier_summary: list[dict[str, Any]] = []
    for tier_name, group in annotated_df.groupby('coverage_tier'):
        y_true = group['true_label'].to_numpy()
        y_prob = group['pred_prob'].to_numpy()
        y_pred = group['binary_pred'].to_numpy()

        acc = accuracy_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5

        # False positive rate and false negative rate
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)

        tier_summary.append({
            'coverage_tier': tier_name,
            'pair_count': len(group),
            'accuracy': float(acc),
            'auroc': float(auroc),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        })

    summary_df = pd.DataFrame(tier_summary)
    summary_df.to_csv(out_p / 'cold_start_coverage_summary.csv', index=False)
    summary_df.to_csv(out_p / 'coverage_tier_report.csv', index=False)

    print(f"\n{'=' * 80}")
    print(f"COLD-START ERROR ANALYSIS BY EXTERNAL PROFILE TIER (Optimal Threshold: {optimal_threshold:.4f}):")
    print(f"{'=' * 80}")
    print(summary_df.to_string(index=False))
    return annotated_df, summary_df


def evaluate_multimodal_calibration(
    model: AuditDDIModel,
    cache: MolecularCache,
    val_df: pd.DataFrame,
    test_splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Measure ECE, fit Platt scaling on transductive validation, and test on cold-start."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    val_loader = _make_dataloader(val_df, cache, batch_size=64, shuffle=False)
    val_probs, val_targets = predict_loader(model, val_loader, device, is_multimodal=True)

    uncal_val_ece = float(expected_calibration_error(val_targets, val_probs, bins=10) or 0.0)
    calibrator = fit_platt_calibrator(val_targets, val_probs)
    cal_val_probs = apply_calibrator(val_probs, calibrator)
    cal_val_ece = float(expected_calibration_error(val_targets, cal_val_probs, bins=10) or 0.0)

    calibration_report: dict[str, Any] = {
        'val_ece_uncalibrated': uncal_val_ece,
        'val_ece_calibrated': cal_val_ece,
        'platt_weights': {
            'w': float(calibrator.get('coefficient', 1.0)),
            'b': float(calibrator.get('intercept', 0.0)),
        },
    }

    print(f"\n{'=' * 80}")
    print("RELIABILITY & CALIBRATION ANALYSIS (ECE):")
    print(f"{'=' * 80}")
    print(f"Validation ECE (Uncalibrated) : {uncal_val_ece:.4f}")
    print(f"Validation ECE (Platt-Scaled) : {cal_val_ece:.4f}")

    from src.models.calibration import fit_temperature_scaling

    # Fit temperature calibrator strictly on held-out validation data (zero test leakage)
    val_temp_cal = fit_temperature_scaling(val_targets, val_probs, fitted_on='validation')
    cal_val_temp_probs = apply_calibrator(val_probs, val_temp_cal)
    cal_val_temp_ece = float(expected_calibration_error(val_targets, cal_val_temp_probs, bins=10) or 0.0)
    print(f"Validation ECE (Temp-Scaled)  : {cal_val_temp_ece:.4f} (T={val_temp_cal.get('temperature', 1.0):.2f})")
    calibration_report['validation_temperature'] = float(val_temp_cal.get('temperature', 1.0))
    calibration_report['validation_ece_temp_calibrated'] = cal_val_temp_ece

    for name, split_df in test_splits.items():
        if split_df.empty:
            continue
        loader = _make_dataloader(split_df, cache, batch_size=64, shuffle=False)
        probs, tgts = predict_loader(model, loader, device, is_multimodal=True)
        uncal_ece = float(expected_calibration_error(tgts, probs, bins=10) or 0.0)
        cal_probs = apply_calibrator(probs, calibrator)
        cal_ece = float(expected_calibration_error(tgts, cal_probs, bins=10) or 0.0)

        # Apply the validation-fitted temperature scaling out-of-sample
        temp_cal_probs = apply_calibrator(probs, val_temp_cal)
        temp_ece = float(expected_calibration_error(tgts, temp_cal_probs, bins=10) or 0.0)

        calibration_report[f'{name}_ece_uncalibrated'] = uncal_ece
        calibration_report[f'{name}_ece_calibrated'] = cal_ece
        calibration_report[f'{name}_temperature_fitted_on'] = 'validation'
        calibration_report[f'{name}_temperature'] = float(val_temp_cal.get('temperature', 1.0))
        calibration_report[f'{name}_ece_temp_calibrated'] = temp_ece
        print(f"Split: {name:<15} | Raw ECE: {uncal_ece:.4f} | Platt ECE: {cal_ece:.4f} | Out-of-Sample Temp ECE: {temp_ece:.4f} (T={val_temp_cal.get('temperature', 1.0):.2f})")

    with open(out_p / 'calibration_report.json', 'w', encoding='utf-8') as f:
        json.dump(calibration_report, f, indent=2)
    with open(out_p / 'calibration_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(calibration_report, f, indent=2)

    return calibration_report


def evaluate_multimodal_subcohort_generalization(
    model: AuditDDIModel,
    cache: MolecularCache,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    device: torch.device | None = None,
    optimal_threshold: float | None = None,
) -> pd.DataFrame:
    """Evaluate generalization across modality-annotated sub-cohorts of the S1 test split.

    Note: These represent modality coverage slices of the S1 cold-start split
    (BindingDB target vs FAERS adverse events vs PharmGKB coverage), not independent
    external datasets.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if test_df.empty or len(test_df) == 0:
        empty_df = pd.DataFrame(columns=['subcohort', 'pair_count', 'auroc', 'auprc', 'accuracy'])
        empty_df.to_csv(out_p / 's1_multimodal_subcohort_profiling.csv', index=False)
        empty_df.to_csv(out_p / 'cross_dataset_generalization_report.csv', index=False)
        return empty_df

    loader = _make_dataloader(test_df, cache, batch_size=64, shuffle=False)
    scores, targets = predict_loader(model, loader, device, is_multimodal=True)

    if optimal_threshold is None:
        try:
            fpr_arr, tpr_arr, thresh_arr = roc_curve(targets, scores)
            j_scores = 2.0 * tpr_arr - fpr_arr
            best_idx = int(np.argmax(j_scores)) if len(j_scores) else 0
            optimal_threshold = float(thresh_arr[best_idx]) if len(thresh_arr) > best_idx else 0.38
            optimal_threshold = max(min(optimal_threshold, 0.45), 0.20)
        except Exception:
            optimal_threshold = 0.38

    src_col = 'drug_a_id' if 'drug_a_id' in test_df.columns else test_df.columns[0]
    tgt_col = 'drug_b_id' if 'drug_b_id' in test_df.columns else test_df.columns[1]

    rows: list[dict[str, Any]] = []
    for idx, (_, r) in enumerate(test_df.iterrows()):
        sa, sb = str(r[src_col]).strip(), str(r[tgt_col]).strip()
        y = float(targets[idx])
        p = float(scores[idx])

        has_target = (cache.target_masks.get(sa, torch.tensor(0.0)).item() > 0.5) and (cache.target_masks.get(sb, torch.tensor(0.0)).item() > 0.5)
        has_faers = (cache.toxicity_masks.get(sa, torch.tensor(0.0)).item() > 0.5) and (cache.toxicity_masks.get(sb, torch.tensor(0.0)).item() > 0.5)
        has_gene = (cache.gene_masks.get(sa, torch.tensor(0.0)).item() > 0.5) and (cache.gene_masks.get(sb, torch.tensor(0.0)).item() > 0.5)
        has_uniprot = bool(cache.target_sequences.get(sa, "")) and bool(cache.target_sequences.get(sb, ""))

        rows.append({
            'drug_a': sa,
            'drug_b': sb,
            'target': y,
            'prob': p,
            'has_bindingdb_target': has_target,
            'has_faers_toxicity': has_faers,
            'has_pharmgkb_gene': has_gene,
            'has_uniprot_target': has_uniprot,
        })

    cols = ['drug_a', 'drug_b', 'target', 'prob', 'has_bindingdb_target', 'has_faers_toxicity', 'has_pharmgkb_gene', 'has_uniprot_target']
    eval_df = pd.DataFrame(rows, columns=cols)

    subsets = [
        ('All S1 Pairs', eval_df),
        ('UniProt Target Sequence Profiled', eval_df[eval_df['has_uniprot_target']]),
        ('BindingDB Target Profiled', eval_df[eval_df['has_bindingdb_target']]),
        ('FAERS Toxicity Profiled', eval_df[eval_df['has_faers_toxicity']]),
        ('PharmGKB Pathway Profiled', eval_df[eval_df['has_pharmgkb_gene']]),
        ('Dual Target + Toxicity Profiled', eval_df[eval_df['has_bindingdb_target'] & eval_df['has_faers_toxicity']]),
    ]

    results: list[dict[str, Any]] = []
    for cohort_name, sub in subsets:
        if len(sub) < 5 or len(sub['target'].unique()) < 2:
            continue
        y_t = sub['target'].to_numpy()
        y_p = sub['prob'].to_numpy()
        auroc = float(roc_auc_score(y_t, y_p))
        auprc = float(average_precision_score(y_t, y_p))
        b_preds = (y_p >= optimal_threshold).astype(int)
        b_acc = float(accuracy_score(y_t, b_preds))
        b_f1 = float(f1_score(y_t, b_preds, zero_division=0))
        pos_m = (y_t == 1)
        b_rec = float(np.sum((b_preds == 1) & pos_m) / max(pos_m.sum(), 1))
        results.append({
            'subcohort': cohort_name,
            'cross_dataset_cohort': cohort_name,
            'pair_count': len(sub),
            'auroc': auroc,
            'auprc': auprc,
            'accuracy': b_acc,
            'f1': b_f1,
            'recall': b_rec,
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_p / 's1_multimodal_subcohort_profiling.csv', index=False)
    res_df.to_csv(out_p / 'cross_dataset_generalization_report.csv', index=False)

    print(f"\n{'=' * 80}")
    print(f"S1 MULTIMODAL SUB-COHORT COVERAGE ANALYSIS (Decision Threshold: {optimal_threshold:.4f}):")
    print(f"{'=' * 80}")
    print(res_df.to_string(index=False))
    return res_df


# Maintain backward-compatible alias for existing pipelines
evaluate_cross_dataset_generalization = evaluate_multimodal_subcohort_generalization


def generate_literature_benchmark_report(
    extended_metrics: dict[str, Any],
    output_dir: Path | str,
    cross_dataset_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Report measured split metrics without unsupported literature targets."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    warm_auroc = float(extended_metrics.get('transductive_auroc', extended_metrics.get('transductive_test_auroc', 0.0)))
    warm_auprc = float(extended_metrics.get('transductive_auprc', 0.0))
    warm_rec = float(extended_metrics.get('transductive_recall', extended_metrics.get('transductive_sensitivity', 0.0)))

    cold_drug_auroc = float(extended_metrics.get('s2_semi_auroc', extended_metrics.get('s2_test_auroc', 0.0)))
    cold_drug_auprc = float(extended_metrics.get('s2_semi_auprc', 0.0))
    cold_drug_rec = float(extended_metrics.get('s2_semi_recall', extended_metrics.get('s2_semi_sensitivity', 0.0)))

    cold_target_auroc = 0.0
    cold_target_auprc = 0.0
    cold_target_rec = 0.0
    if cross_dataset_df is not None and not cross_dataset_df.empty:
        c_col = 'subcohort' if 'subcohort' in cross_dataset_df.columns else 'cross_dataset_cohort'
        primary_cohorts = cross_dataset_df[cross_dataset_df[c_col].isin(['PharmGKB Pathway Profiled', 'BindingDB Target Profiled'])]
        if not primary_cohorts.empty:
            cold_target_auroc = float(primary_cohorts['auroc'].mean())
            cold_target_auprc = float(primary_cohorts['auprc'].mean())
            cold_target_rec = float(primary_cohorts.get('recall', pd.Series([0.0])).mean())
        else:
            prof_rows = cross_dataset_df[cross_dataset_df[c_col].astype(str).str.contains('Target|Pathway', case=False, na=False)]
            if not prof_rows.empty:
                cold_target_auroc = float(prof_rows['auroc'].mean())
                cold_target_auprc = float(prof_rows['auprc'].mean())
                cold_target_rec = float(prof_rows.get('recall', pd.Series([0.0])).mean())

    s1_auroc = float(extended_metrics.get('s1_best_auroc', extended_metrics.get('s1_cold_auroc', extended_metrics.get('s1_test_auroc', 0.0))))
    s1_auprc = float(extended_metrics.get('s1_best_auprc', extended_metrics.get('s1_cold_auprc', 0.0)))
    s1_rec = float(extended_metrics.get('s1_best_recall', extended_metrics.get('s1_cold_recall', extended_metrics.get('s1_best_sensitivity', 0.0))))

    rows = [
        {
            'Evaluation split': 'Transductive',
            'Definition': 'Both drugs occur in the training drug set; pair held out.',
            'AUROC': f"{warm_auroc:.4f}",
            'AUPRC': f"{warm_auprc:.4f}",
            'Recall at validation threshold': f"{warm_rec*100:.1f}%",
        },
        {
            'Evaluation split': 'S2',
            'Definition': 'Exactly one drug is absent from the training drug set.',
            'AUROC': f"{cold_drug_auroc:.4f}",
            'AUPRC': f"{cold_drug_auprc:.4f}",
            'Recall at validation threshold': f"{cold_drug_rec*100:.1f}%",
        },
        {
            'Evaluation split': 'S1 with target/pathway coverage',
            'Definition': 'S1 subgroup analysis by available biological annotations; not a separate cold-target split.',
            'AUROC': f"{cold_target_auroc:.4f}" if cold_target_auroc > 0 else 'N/A',
            'AUPRC': f"{cold_target_auprc:.4f}" if cold_target_auprc > 0 else 'N/A',
            'Recall at validation threshold': f"{cold_target_rec*100:.1f}%" if cold_target_rec > 0 else 'N/A',
        },
        {
            'Evaluation split': 'S1',
            'Definition': 'Neither drug occurs in the training drug set.',
            'AUROC': f"{s1_auroc:.4f}",
            'AUPRC': f"{s1_auprc:.4f}",
            'Recall at validation threshold': f"{s1_rec*100:.1f}%",
        },
    ]

    bench_df = pd.DataFrame(rows)
    bench_df.to_csv(out_p / 'literature_benchmark_comparison.csv', index=False)

    md_lines = [
        "# Measured split performance",
        "",
        "Thresholded recall uses the threshold selected on validation data. These are single-run measurements, not clinical performance claims.",
        "",
        "| Evaluation split | Definition | AUROC | AUPRC | Recall at validation threshold |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(f"| {r['Evaluation split']} | {r['Definition']} | {r['AUROC']} | {r['AUPRC']} | {r['Recall at validation threshold']} |")
    md_lines.append('')
    (out_p / 'literature_benchmark_comparison.md').write_text('\n'.join(md_lines), encoding='utf-8')
    print('\nMeasured transductive/S1/S2 and S1 coverage-subgroup results saved; no fixed literature target is assumed.')

    return bench_df


def resolve_existing_dir(candidates: list[Path | str | None]) -> Path | None:
    """Return the first candidate path that exists as a directory, case-insensitively."""
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.is_dir():
            return p
    # Case-insensitive search
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        parent = p.parent
        target_name = p.name.lower()
        if parent.is_dir():
            try:
                for child in parent.iterdir():
                    if child.is_dir() and child.name.lower() == target_name:
                        return child
            except Exception:
                pass
    return None


def resolve_existing_path(candidates: list[Path | str | None]) -> Path | None:
    """Return the first candidate path that exists (file or dir), case-insensitively."""
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.exists():
            return p
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        parent = p.parent
        target_name = p.name.lower()
        if parent.is_dir():
            try:
                for child in parent.iterdir():
                    if child.name.lower() == target_name:
                        return child
            except Exception:
                pass
    return None


def run_full_multimodal_study(
    master_nodes_path: str | Path | None = None,
    splits_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    master_edges_path: str | Path | None = None,
    extended_epochs: int = 15,
    ablation_epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    device: torch.device | None = None,
    pretrained_encoder_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute complete extended training, ablation study, error analysis, and calibration."""
    if master_nodes_path is None:
        master_nodes_path = kwargs.pop('master_nodes_csv', None)
    if master_nodes_path is None:
        raise ValueError('master_nodes_path (or master_nodes_csv) must be provided')

    # Support flexible parameter aliases
    if 'epochs' in kwargs:
        extended_epochs = int(kwargs.pop('epochs'))
    if 'lr' in kwargs:
        learning_rate = float(kwargs.pop('lr'))
    run_ablation: bool = kwargs.pop('run_ablation', True)
    run_error_analysis: bool = kwargs.pop('run_error_analysis', True)
    calibrate: bool = kwargs.pop('calibrate', True)
    pos_weight: float = float(kwargs.pop('pos_weight', 2.2))
    use_ssl: bool = bool(kwargs.pop('use_ssl', False))
    use_faers_features: bool = bool(kwargs.pop('use_faers_features', False))
    if use_faers_features:
        raise ValueError(
            'The current FAERS bridge has no verified as-of date; full-history toxicity '
            'scores cannot be used as DDI model inputs without a temporal leakage audit.'
        )
    ssl_weight: float = float(kwargs.pop('ssl_weight', 0.2))
    memory_dropout: float = float(kwargs.pop('memory_dropout', 0.75))
    bio_align_weight: float = float(kwargs.pop('bio_align_weight', 0.25))
    embedding_noise_std: float = float(kwargs.pop('embedding_noise_std', 0.02))
    patience: int = int(kwargs.pop('patience', 8))
    model_seed = int(kwargs.pop('seed', os.environ.get('AUDITDDI_MODEL_SEED', '42')))
    split_seed = int(kwargs.pop('split_seed', os.environ.get('AUDITDDI_SPLIT_SEED', '42')))
    if model_seed <= 0 or split_seed <= 0:
        raise ValueError('model_seed and split_seed must be positive integers.')
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    if output_dir is None:
        try:
            from src.data_prep.path_resolver import resolve_results_base
            out_p = resolve_results_base() / 'multimodal_study_results'
        except Exception:
            out_p = Path(master_nodes_path).resolve().parent.parent / 'multimodal_study_results'
    else:
        out_p = Path(output_dir)
    # Callers can assign independent output folders per seed; the dedicated
    # Colab launcher does this so separate accounts never overwrite artifacts.
    out_p.mkdir(parents=True, exist_ok=True)

    splits_p = ensure_benchmark_splits(
        splits_dir=splits_dir,
        master_nodes_path=master_nodes_path,
        master_edges_path=master_edges_path,
        seed=split_seed,
        **kwargs,
    )

    # Resolve candidate dataset roots
    resolved_nodes = Path(master_nodes_path).resolve()
    try:
        from src.data_prep.path_resolver import resolve_data_base
        canonical_base = resolve_data_base()
    except Exception:
        canonical_base = None

    candidate_data_roots = [
        resolved_nodes.parent.parent,
        resolved_nodes.parent,
    ]
    if canonical_base:
        candidate_data_roots.insert(0, canonical_base)
    candidate_data_roots.extend([
        Path('/content/drive/MyDrive/auditddi-data'),
        Path('/content/drive/MyDrive/auditddi'),
        Path('/content/drive/MyDrive/pxddi-data'),
        Path('auditddi-data'),
        Path('pxddi-data'),
        Path('.'),
    ])
    if 'data_dir' in kwargs and kwargs['data_dir']:
        candidate_data_roots.insert(0, Path(kwargs.pop('data_dir')).resolve())

    data_root = next((r for r in candidate_data_roots if r.is_dir()), resolved_nodes.parent.parent)

    print("=" * 80)
    print("STARTING AUDITDDI MULTIMODAL COMPREHENSIVE STUDY")
    print(f"Data Root    : {data_root}")
    print(f"Master Nodes : {master_nodes_path}")
    print(f"Splits Dir   : {splits_p}")
    print(f"Output Dir   : {out_p}")
    print("=" * 80)

    # 1. Auto-enrich master nodes with PharmGKB pharmacogenomic pathways first (populates gene_symbols)
    cand_pharmgkb = resolve_existing_dir([
        kwargs.pop('pharmgkb_dir', None),
        data_root / 'pharmgkb',
        data_root / 'PharmGKB',
        resolved_nodes.parent / 'pharmgkb',
    ])
    if cand_pharmgkb:
        try:
            sample_df = pd.read_csv(master_nodes_path, nrows=10)
            if 'gene_vector_multihot' not in sample_df.columns or sample_df['gene_vector_multihot'].dropna().empty:
                from src.data_prep.pharmgkb_pipeline import update_master_nodes_with_pharmgkb_pathways
                from src.data_prep.expanded_pharmgkb_bridge import update_master_nodes_with_pharmgkb_faers_analogs
                print(f"Auto-enriching master nodes with PharmGKB pathways from: {cand_pharmgkb}")
                update_master_nodes_with_pharmgkb_pathways(master_nodes_path, cand_pharmgkb)
                update_master_nodes_with_pharmgkb_faers_analogs(master_nodes_path)
        except Exception as pgkb_err:
            print(f"PharmGKB auto-enrichment notice: {pgkb_err}")

    # 2. Auto-enrich master nodes with FAERS clinical toxicity
    cand_faers = resolve_existing_path([
        kwargs.pop('faers_dir', None),
        kwargs.pop('faers_bridge_path', None),
        data_root / 'faers' / 'faers_bridge.csv',
        data_root / 'faers',
        data_root / 'FAERS',
        resolved_nodes.parent / 'faers_bridge.csv',
        resolved_nodes.parent / 'faers',
    ])
    if cand_faers:
        try:
            sample_df = pd.read_csv(master_nodes_path, nrows=10)
            if 'toxicity_score' not in sample_df.columns or sample_df['toxicity_score'].dropna().empty:
                from src.data_prep.build_unified_graph import update_master_nodes_with_faers
                print(f"Auto-enriching master nodes with FAERS from: {cand_faers}")
                update_master_nodes_with_faers(master_nodes_path, cand_faers)
        except Exception as faers_err:
            print(f"FAERS auto-enrichment notice: {faers_err}")

    # 3. Auto-enrich master nodes with BindingDB, GEO, and PDB (PDB now has rich gene symbols and targets available)
    for mod_name, dir_key, col_names, enrich_fn in [
        ('BindingDB', 'bindingdb_dir', ['bindingdb_target_vector', 'target_vector_multihot'], 'src.data_prep.bindingdb_pipeline.update_master_nodes_with_bindingdb'),
        ('GEO', 'geo_dir', ['geo_signature_vector', 'geo_vector'], 'src.data_prep.geo_pipeline.update_master_nodes_with_geo'),
        ('PDB', 'pdb_dir', ['pdb_vector_multihot', 'pdb_vector'], 'src.data_prep.pdb_pipeline.update_master_nodes_with_pdb'),
    ]:
        cand_dir = resolve_existing_dir([
            kwargs.pop(dir_key, None),
            data_root / mod_name,
            data_root / mod_name.lower(),
            resolved_nodes.parent / mod_name,
            resolved_nodes.parent / mod_name.lower(),
        ])
        if cand_dir:
            try:
                sample_df = pd.read_csv(master_nodes_path)
                needs_enrichment = False
                found_cols = [c for c in col_names if c in sample_df.columns]
                if not found_cols:
                    needs_enrichment = True
                else:
                    col_data = sample_df[found_cols[0]].dropna()
                    if col_data.empty:
                        needs_enrichment = True
                    else:
                        has_nonzero = False
                        for val_str in col_data.iloc[:100]:
                            try:
                                parsed = json.loads(val_str) if isinstance(val_str, str) else list(val_str)
                                if any(x != 0 for x in parsed):
                                    has_nonzero = True
                                    break
                            except Exception:
                                pass
                        if not has_nonzero:
                            needs_enrichment = True
                if mod_name == 'PDB' and 'is_pdb_active' in sample_df.columns:
                    if int(sample_df['is_pdb_active'].sum()) == 0:
                        needs_enrichment = True
                if needs_enrichment:
                    mod_path, fn_name = enrich_fn.rsplit('.', 1)
                    module = __import__(mod_path, fromlist=[fn_name])
                    fn = getattr(module, fn_name)
                    print(f"Auto-enriching master nodes with {mod_name} from: {cand_dir}")
                    fn(master_nodes_path, cand_dir)
            except Exception as enrich_err:
                print(f"{mod_name} auto-enrichment notice: {enrich_err}")

    # 1. Populate Cache
    cache = MolecularCache(gene_dim=50)
    cache.populate_from_master_nodes(master_nodes_path)

    # 3. Load Splits
    train_df = pd.read_csv(splits_p / 'transductive_train.csv')
    val_df = pd.read_csv(splits_p / 'validation.csv')
    test_splits = {
        'transductive': pd.read_csv(splits_p / 'transductive_test.csv'),
        's1_cold': pd.read_csv(splits_p / 's1_test.csv'),
        's2_semi': pd.read_csv(splits_p / 's2_test.csv'),
    }
    cold_dev_splits = {
        's1_dev': pd.read_csv(splits_p / 's1_dev.csv'),
        's2_dev': pd.read_csv(splits_p / 's2_dev.csv'),
    }

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    split_hashes = {
        path.name: sha256_file(path)
        for path in sorted(splits_p.glob('*.csv'))
        if path.name in {'transductive_train.csv', 'validation.csv', 'transductive_test.csv', 's1_dev.csv', 's2_dev.csv', 's1_test.csv', 's2_test.csv'}
    }
    input_manifest = {
        'model_seed': model_seed,
        'split_seed': split_seed,
        'master_nodes_path': str(Path(master_nodes_path).resolve()),
        'master_nodes_sha256': sha256_file(Path(master_nodes_path)),
        'split_directory': str(splits_p.resolve()),
        'split_sha256': split_hashes,
        'epochs': extended_epochs,
        'batch_size': batch_size,
        'evaluation_policy': 'checkpoint_selected_by_macro_auroc_on_transductive_validation_s1_dev_s2_dev; per_split_thresholds_selected_on_matching_dev; test_only_after_selection',
    }
    (out_p / 'run_input_manifest.json').write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True), encoding='utf-8'
    )

    chembl_pretrained_path: str | Path | None = (
        pretrained_encoder_path
        or kwargs.pop('chembl_pretrained_path', None)
        or kwargs.pop('chembl_encoder_checkpoint', None)
        or kwargs.pop('pretrained_checkpoint', None)
        or kwargs.pop('encoder_checkpoint', None)
        or kwargs.pop('chembl_checkpoint', None)
    )
    if chembl_pretrained_path is not None:
        raise ValueError(
            'The multimodal split artifacts do not match the standalone ChEMBL '
            'pretraining split contract. Run ChEMBL as a separately audited '
            'candidate with src/training/run_experiment_suite.py.'
        )

    print("\n" + "=" * 80)
    print("AUDITDDI NINE-SOURCE MULTIMODAL INGESTION SUMMARY:")
    print("=" * 80)
    print(f"[1/9] TWOSIDES : Ground-truth DDI labels ({len(train_df):,} train, {len(val_df):,} validation, {len(test_splits['s1_cold']):,} S1 test pairs)")
    chembl_status = 'Not used in this candidate: its checkpoint is evaluated separately under the matching split-provenance protocol.'
    print(f"[2/9] ChEMBL   : {chembl_status}")
    print('[3/9] PubChem  : Identity/structure cross-reference only where present in the prepared master catalog.')
    n_genes = sum(1 for m in cache.gene_masks.values() if m.item() > 0)
    print(f"[4/9] PharmGKB : CYP/enzyme gene profiles ({n_genes}/{len(cache.graphs)} drugs, dim={cache.gene_dim})")
    n_targets = sum(1 for m in cache.target_masks.values() if m.item() > 0)
    print(f"[5/9] BindingDB: Drug-target profiles ({n_targets}/{len(cache.graphs)} drugs, dim={cache.target_dim})")
    n_sequences = sum(bool(sequence) for sequence in cache.target_sequences.values())
    print(f"[6/9] UniProt  : Target sequences ({n_sequences}/{len(cache.graphs)} drugs)")
    n_geo = sum(1 for m in cache.geo_masks.values() if m.item() > 0)
    print(f"[7/9] GEO      : Perturbation profiles ({n_geo}/{len(cache.graphs)} drugs, dim={cache.geo_dim})")
    n_tox = sum(1 for m in cache.toxicity_masks.values() if m.item() > 0)
    print(f"[8/9] FAERS    : {n_tox}/{len(cache.graphs)} toxicity records audited; full-history scores disabled as DDI input pending time-cutoff/overlap audit.")
    n_pdb = sum(1 for m in cache.pdb_masks.values() if m.item() > 0)
    print(f"[9/9] PDB      : Protein-ligand target profiles ({n_pdb}/{len(cache.graphs)} drugs, dim={cache.pdb_dim})")
    print(f"Derived chemical inputs: RDKit molecular graphs and Morgan fingerprints ({len(cache.graphs):,} structures).")
    print("=" * 80 + "\n")

    def split_pair_coverage(frame: pd.DataFrame, masks: dict[str, torch.Tensor]) -> dict[str, Any]:
        left_col = 'drug_a_id' if 'drug_a_id' in frame.columns else 'source'
        right_col = 'drug_b_id' if 'drug_b_id' in frame.columns else 'target'
        covered_both = 0
        covered_either = 0
        for left, right in zip(frame[left_col].astype(str), frame[right_col].astype(str)):
            left_ok = masks.get(left, torch.tensor(0.0)).item() > 0.5
            right_ok = masks.get(right, torch.tensor(0.0)).item() > 0.5
            covered_both += int(left_ok and right_ok)
            covered_either += int(left_ok or right_ok)
        total = len(frame)
        return {
            'pair_rows': total,
            'both_drugs_covered': covered_both,
            'either_drug_covered': covered_either,
            'both_coverage_fraction': covered_both / total if total else None,
        }

    frames_by_split = {
        'train': train_df,
        'validation': val_df,
        'transductive_test': test_splits['transductive'],
        's1_test': test_splits['s1_cold'],
        's2_test': test_splits['s2_semi'],
    }
    input_manifest.update({
        'chembl_pretrained_checkpoint': (
            str(Path(chembl_pretrained_path).resolve())
            if chembl_pretrained_path and Path(chembl_pretrained_path).is_file()
            else None
        ),
        'chembl_pretrained_checkpoint_sha256': (
            sha256_file(Path(chembl_pretrained_path))
            if chembl_pretrained_path and Path(chembl_pretrained_path).is_file()
            else None
        ),
        'resolved_data_root': str(data_root.resolve()),
        'feature_coverage_drugs': {
            'molecular_graph': len(cache.graphs),
            'rdkit_morgan_fingerprint': len(cache.fingerprints),
            'pharmgkb_gene': n_genes,
            'faers_toxicity': n_tox,
            'bindingdb_target': n_targets,
            'geo_expression': n_geo,
            'pdb_structure': n_pdb,
            'uniprot_sequence': sum(bool(sequence) for sequence in cache.target_sequences.values()),
        },
        'feature_dimensions': {
            'pharmgkb_gene': cache.gene_dim,
            'bindingdb_target': cache.target_dim,
            'geo_expression': cache.geo_dim,
            'pdb_structure': cache.pdb_dim,
        },
        'pair_coverage_by_split': {
            'pharmgkb_gene': {
                name: split_pair_coverage(frame, cache.gene_masks)
                for name, frame in frames_by_split.items()
            },
            'faers_toxicity_audited_but_not_used': {
                name: split_pair_coverage(frame, cache.toxicity_masks)
                for name, frame in frames_by_split.items()
            },
            'bindingdb_target': {
                name: split_pair_coverage(frame, cache.target_masks)
                for name, frame in frames_by_split.items()
            },
            'geo_expression': {
                name: split_pair_coverage(frame, cache.geo_masks)
                for name, frame in frames_by_split.items()
            },
            'pdb_structure': {
                name: split_pair_coverage(frame, cache.pdb_masks)
                for name, frame in frames_by_split.items()
            },
            'uniprot_sequence': {
                name: split_pair_coverage(
                    frame,
                    {drug: torch.tensor(float(bool(seq))) for drug, seq in cache.target_sequences.items()},
                )
                for name, frame in frames_by_split.items()
            },
        },
    })
    (out_p / 'run_input_manifest.json').write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True), encoding='utf-8'
    )

    use_cross_modal_attention: bool = kwargs.pop('use_cross_modal_attention', True)
    use_cross_drug_attention: bool = kwargs.pop('use_cross_drug_attention', False)
    use_target_encoder: bool = kwargs.pop('use_target_encoder', True)
    use_protein_sequence_encoder: bool = bool(kwargs.pop('use_protein_sequence_encoder', True))
    use_neighbor_memory: bool = kwargs.pop('use_neighbor_memory', False)
    select_best_by: str = kwargs.pop('select_best_by', 'val')

    neighbor_mem = None
    if use_neighbor_memory:
        from src.models.neighbor_memory import AuditableNeighborMemory
        src_col = 'drug_a_id' if 'drug_a_id' in train_df.columns else ('source' if 'source' in train_df.columns else train_df.columns[0])
        tgt_col = 'drug_b_id' if 'drug_b_id' in train_df.columns else ('target' if 'target' in train_df.columns else train_df.columns[1])
        lbl_col = 'label' if 'label' in train_df.columns else train_df.columns[-1]
        neighbor_mem = AuditableNeighborMemory(k_neighbors=5)
        neighbor_mem.fit(train_df[src_col].tolist(), train_df[tgt_col].tolist(), train_df[lbl_col].tolist())

    # 4. Extended Training (Full Multimodal Model)
    best_model, history_df, extended_metrics = train_extended_multimodal(
        cache=cache,
        train_df=train_df,
        val_df=val_df,
        test_splits=test_splits,
        cold_dev_splits=cold_dev_splits,
        output_dir=out_p,
        epochs=extended_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        chembl_pretrained_path=chembl_pretrained_path,
        use_cross_modal_attention=use_cross_modal_attention,
        use_cross_drug_attention=use_cross_drug_attention,
        use_target_encoder=use_target_encoder,
        use_faers_features=use_faers_features,
        use_protein_sequence_encoder=use_protein_sequence_encoder,
        use_neighbor_memory=use_neighbor_memory,
        select_best_by=select_best_by,
        pos_weight=pos_weight,
        use_ssl=use_ssl,
        ssl_weight=ssl_weight,
        memory_dropout=memory_dropout,
        embedding_noise_std=embedding_noise_std,
        bio_align_weight=bio_align_weight,
        patience=patience,
    )

    # 5. Modality Ablation Study
    ablation_dict: list[dict[str, Any]] = []
    if run_ablation:
        ablation_df = run_modality_ablation_study(
            cache=cache,
            train_df=train_df,
            val_df=val_df,
            test_splits=test_splits,
            cold_dev_splits=cold_dev_splits,
            output_dir=out_p / 'ablation',
            epochs=ablation_epochs,
            batch_size=batch_size,
            device=device,
            chembl_pretrained_path=chembl_pretrained_path,
            use_neighbor_memory=use_neighbor_memory,
            select_best_by=select_best_by,
            pos_weight=pos_weight,
            use_ssl=use_ssl,
            use_target_encoder=use_target_encoder,
            use_faers_features=use_faers_features,
            memory_dropout=memory_dropout,
        )
        ablation_dict = cast(list[dict[str, Any]], ablation_df.to_dict(orient='records'))

    # 6. Cold-Start Error Analysis Stratified by External Coverage
    tier_dict: list[dict[str, Any]] = []
    opt_thresh_for_eval = extended_metrics.get('val_optimal_threshold', extended_metrics.get('s1_best_opt_thresh', extended_metrics.get('s1_cold_optimal_threshold', extended_metrics.get('s1_cold_opt_thresh', 0.35))))
    if run_error_analysis:
        err_df, tier_summary_df = analyze_cold_start_coverage_errors(
            model=best_model,
            cache=cache,
            s1_test_df=test_splits['s1_cold'],
            output_dir=out_p / 'error_analysis',
            device=device,
            neighbor_memory=neighbor_mem,
            optimal_threshold=opt_thresh_for_eval,
        )
        tier_dict = cast(list[dict[str, Any]], tier_summary_df.to_dict(orient='records'))

    # 6. Model Calibration (ECE & Reliability)
    calibration_report: dict[str, Any] = {}
    if calibrate:
        calibration_report = evaluate_multimodal_calibration(
            model=best_model,
            cache=cache,
            val_df=val_df,
            test_splits=test_splits,
            output_dir=out_p / 'calibration',
            device=device,
        )

    # 7. Cross-Dataset Validation
    cross_dataset_df: pd.DataFrame | None = None
    cross_dataset_dict: list[dict[str, Any]] = []
    if 's1_cold' in test_splits:
        cross_dataset_df = evaluate_cross_dataset_generalization(
            model=best_model,
            cache=cache,
            test_df=test_splits['s1_cold'],
            output_dir=out_p / 'cross_dataset',
            device=device,
            optimal_threshold=opt_thresh_for_eval,
        )
        cross_dataset_dict = cast(list[dict[str, Any]], cross_dataset_df.to_dict(orient='records'))

    # 8. Standardized Literature Benchmark Comparison Report
    literature_df = generate_literature_benchmark_report(
        extended_metrics=extended_metrics,
        output_dir=out_p,
        cross_dataset_df=cross_dataset_df if 's1_cold' in test_splits else None,
    )
    literature_dict = cast(list[dict[str, Any]], literature_df.to_dict(orient='records'))

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MULTIMODAL STUDY COMPLETE!")
    print(f"All models, ablation reports, and error analysis saved to: {out_p}")
    print("=" * 80)

    # Format return dictionary to support all client conventions
    transductive_test_auroc = extended_metrics.get('transductive_auroc', extended_metrics.get('transductive_test_auroc', 0.0))
    s1_auroc = extended_metrics.get('s1_cold_auroc', extended_metrics.get('s1_test_auroc', 0.0))
    s2_auroc = extended_metrics.get('s2_semi_auroc', extended_metrics.get('s2_test_auroc', 0.0))
    peak_s1_auroc = s1_auroc

    transductive_and_cold_metrics = {
        'test_auroc': transductive_test_auroc,
        's1_cold_auroc': s1_auroc,
        's2_semi_auroc': s2_auroc,
        **extended_metrics,
    }
    peak_s1_metrics = {
        's1_cold_auroc': peak_s1_auroc,
        'test_auroc': peak_s1_auroc,
        'evaluation_note': 'S1 is evaluated once at the validation-selected checkpoint; no peak-test selection.',
        's1_auprc': extended_metrics.get('s1_cold_average_precision', extended_metrics.get('s1_cold_auprc')),
    }

    return {
        'extended_metrics': extended_metrics,
        'transductive_and_cold_metrics': transductive_and_cold_metrics,
        'peak_s1_metrics': peak_s1_metrics,
        'ablation': ablation_dict,
        'ablation_results': ablation_dict,
        'tier_summary': tier_dict,
        'calibration_report': calibration_report,
        'cross_dataset_validation': cross_dataset_dict,
        'literature_benchmark': literature_dict,
    }


# Convenience alias matching external call conventions
run_full_study = run_full_multimodal_study
