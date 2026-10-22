"""Seven-branch, symmetric binary DDI model for the proposed AuditDDI system.

Each drug is represented by a molecular graph, Morgan fingerprint, drug
profile, adverse-event profile, target profile, 3D structure profile, and
context-defined gene-expression profile.  Vector branches are supplied as
precomputed, CID/SMILES-mapped features with explicit availability masks.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from .encoder import EdgeAwareMolecularEncoder


SEVEN_BRANCH_ARCHITECTURE = 'auditddi_seven_branch_v1'
SEVEN_BRANCH_NAMES = (
    'drug_profile',
    'adverse_event',
    'protein_target',
    'pdb_structure',
    'gene_expression',
)


class _MaskedFeatureEncoder(nn.Module):
    """Encode a drug vector and preserve its observed/missing indicator."""

    def __init__(self, input_dim: int, output_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        if input_dim <= 0:
            raise ValueError('Every seven-branch feature dimension must be positive.')
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.float().view(-1, self.input_dim)
        mask = mask.float().view(-1, 1).clamp(0.0, 1.0)
        return torch.cat((self.network(values) * mask, mask), dim=1)


class SevenBranchDDIModel(nn.Module):
    """Shared per-drug encoders with order-invariant pair fusion.

    Each ordered pair is fused as `[hA, hB, hA - hB]`, as shown in the
    proposed system. The classifier scores both pair orders and averages the
    logits, so swapping drug order cannot change the undirected association.
    """

    def __init__(
        self,
        in_channels: int,
        edge_feature_dim: int,
        hidden_channels: int,
        branch_dimensions: Mapping[str, int],
        fingerprint_dim: int = 1024,
        branch_hidden_channels: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        missing = set(SEVEN_BRANCH_NAMES).difference(branch_dimensions)
        if missing:
            raise ValueError(f'Missing seven-branch dimensions: {sorted(missing)}')
        self.architecture_version = SEVEN_BRANCH_ARCHITECTURE
        self.in_channels = int(in_channels)
        self.edge_feature_dim = int(edge_feature_dim)
        self.hidden_channels = int(hidden_channels)
        self.fingerprint_dim = int(fingerprint_dim)
        self.branch_dimensions = {
            name: int(branch_dimensions[name]) for name in SEVEN_BRANCH_NAMES
        }

        self.encoder = EdgeAwareMolecularEncoder(
            self.in_channels, self.edge_feature_dim, self.hidden_channels
        )
        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(self.fingerprint_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.branch_encoders = nn.ModuleDict({
            name: _MaskedFeatureEncoder(dim, branch_hidden_channels, dropout)
            for name, dim in self.branch_dimensions.items()
        })

        # Graph + fingerprint + (five vector encodings, each with mask bit).
        self.per_drug_dim = (
            self.hidden_channels + 128
            + len(SEVEN_BRANCH_NAMES) * (branch_hidden_channels + 1)
        )
        self.risk_classifier = nn.Sequential(
            nn.LayerNorm(self.per_drug_dim * 3),
            nn.Linear(self.per_drug_dim * 3, 256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _required_graph_tensor(
        graph, name: str, batch_size: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = getattr(graph, f'{name}_features', None)
        mask = getattr(graph, f'{name}_mask', None)
        if value is None or mask is None:
            raise ValueError(
                f'The seven-branch run requires {name}_features and {name}_mask '
                'on every batched drug graph.'
            )
        value = value.float().view(batch_size, -1)
        if value.shape[1] != width:
            raise ValueError(
                f'{name} width mismatch: expected {width}, received {value.shape[1]}.'
            )
        return value, mask.float().view(batch_size, 1)

    def encode_drug(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        molecular = self.encoder(
            graph.x, graph.edge_index, graph.edge_attr, graph.batch
        )
        fingerprint = getattr(graph, 'fingerprint_features', None)
        if fingerprint is None:
            raise ValueError('Seven-branch model requires Morgan fingerprint_features.')
        fingerprint = fingerprint.float().view(molecular.size(0), self.fingerprint_dim)
        parts = [molecular, self.fingerprint_encoder(fingerprint)]
        for name in SEVEN_BRANCH_NAMES:
            values, mask = self._required_graph_tensor(
                graph, name, molecular.size(0), self.branch_dimensions[name]
            )
            parts.append(self.branch_encoders[name](values, mask))
        return torch.cat(parts, dim=1), molecular

    def forward(self, drug_a, drug_b, **_unused):
        representation_a, _ = self.encode_drug(drug_a)
        representation_b, _ = self.encode_drug(drug_b)
        pair_ab = torch.cat((
            representation_a,
            representation_b,
            representation_a - representation_b,
        ), dim=1)
        pair_ba = torch.cat((
            representation_b,
            representation_a,
            representation_b - representation_a,
        ), dim=1)
        risk_logits = 0.5 * (
            self.risk_classifier(pair_ab).squeeze(-1)
            + self.risk_classifier(pair_ba).squeeze(-1)
        )
        # Preserve the legacy trainer's three-tensor return contract. This
        # architecture does not train or claim a drug-toxicity prediction head.
        no_toxicity_prediction = torch.zeros_like(risk_logits)
        return (
            risk_logits,
            no_toxicity_prediction,
            no_toxicity_prediction,
        )

