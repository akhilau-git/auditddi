from backend.compound_registry import CompoundRegistry


def test_compound_registry_persists_review_metadata(tmp_path):
    registry = CompoundRegistry(tmp_path)
    created = registry.add(
        name='Acetaminophen candidate',
        external_id='TEST-001',
        smiles='CC(=O)NC1=CC=C(C=C1)O',
        evidence_notes='Verified from a controlled test source.',
    )

    assert created['id'] > 0
    assert created['external_id'] == 'TEST-001'
    assert registry.list()[0]['name'] == 'Acetaminophen candidate'
