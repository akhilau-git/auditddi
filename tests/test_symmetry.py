"""Regression tests for the order-independent AuditDDI pair architecture."""

from pathlib import Path
import sys

import pytest
import torch
from torch_geometric.data import Batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_prep.prepare_twosides import smiles_to_graph
from models.ddi_model import AuditDDIModel


CHECKPOINT_PATH = PROJECT_ROOT / 'backend' / 'checkpoints' / 'auditddi_model.pt'
TEST_PAIRS = [
    (
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CC(=O)NC1=CC=C(C=C1)O',
    ),
    (
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    ),
]


@pytest.fixture(scope='module')
def model():
    if not CHECKPOINT_PATH.is_file():
        pytest.skip(f"Shipped model checkpoint not found at {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
    loaded_model = AuditDDIModel(
        in_channels=checkpoint['in_channels'],
        hidden_channels=checkpoint['hidden_channels'],
        use_chemberta=checkpoint.get('use_chemberta', False),
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    return loaded_model


@pytest.mark.parametrize(('smiles_a', 'smiles_b'), TEST_PAIRS)
def test_shipped_model_is_order_independent(model, smiles_a, smiles_b):
    """The same pair must produce the same risk score in either input order."""
    graph_a = smiles_to_graph(smiles_a)
    graph_b = smiles_to_graph(smiles_b)
    assert graph_a is not None
    assert graph_b is not None

    batch_a = Batch.from_data_list([graph_a])
    batch_b = Batch.from_data_list([graph_b])

    with torch.no_grad():
        risk_ab, _, _ = model(batch_a, batch_b)
        risk_ba, _, _ = model(batch_b, batch_a)

    difference = abs(
        torch.sigmoid(risk_ab).item() - torch.sigmoid(risk_ba).item()
    )
    assert difference < 1e-6, f'Model is order-sensitive: diff={difference:.8f}'


@pytest.mark.parametrize(('smiles_a', 'smiles_b'), TEST_PAIRS)
def test_initialized_model_is_order_independent(smiles_a, smiles_b):
    """Pair architecture must produce identical scores regardless of drug argument order."""
    graph_a = smiles_to_graph(smiles_a)
    graph_b = smiles_to_graph(smiles_b)
    assert graph_a is not None and graph_b is not None
    assert graph_a.x is not None
    init_model = AuditDDIModel(in_channels=graph_a.x.size(-1), hidden_channels=16).eval()
    batch_a = Batch.from_data_list([graph_a])
    batch_b = Batch.from_data_list([graph_b])
    with torch.no_grad():
        risk_ab, _, _ = init_model(batch_a, batch_b)
        risk_ba, _, _ = init_model(batch_b, batch_a)
    diff = abs(torch.sigmoid(risk_ab).item() - torch.sigmoid(risk_ba).item())
    assert diff < 1e-6, f'Initialized model is order-sensitive: diff={diff:.8f}'

