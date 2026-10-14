"""Tests for path_resolver.py alias resolution and finalized active datasets."""

from pathlib import Path
import pytest

from src.data_prep.path_resolver import (
    ACTIVE_DATASETS,
    DATASET_ALIASES,
    resolve_dataset_subpath,
)


def test_active_datasets_contains_finalized_seven():
    expected_seven = {"twosides", "pubchem", "chembl", "pharmgkb", "ddinter", "openfda", "faers"}
    actual = {ds.lower() for ds in ACTIVE_DATASETS}
    assert expected_seven.issubset(actual)


def test_resolve_dataset_subpath_handles_spelling_aliases(tmp_path: Path):
    # Simulate user's exact folder setup:
    # nsides, chembel, faeres, pharmagkb, pubchem, ddinter, openfda
    (tmp_path / "nsides" / "twosides").mkdir(parents=True)
    (tmp_path / "chembel").mkdir()
    (tmp_path / "faeres").mkdir()
    (tmp_path / "pharmagkb").mkdir()
    (tmp_path / "pubchem").mkdir()
    (tmp_path / "ddinter").mkdir()
    (tmp_path / "openfda").mkdir()

    # Put a dummy file in each to test file resolution
    (tmp_path / "chembel" / "structures.smiles.tsv").write_text("ID\tSMILES\n")
    (tmp_path / "nsides" / "twosides" / "drug_drug_edges.csv").write_text("source,target\n")

    # Canonical queries should find the aliased folders seamlessly
    assert resolve_dataset_subpath(tmp_path, "chembl").name == "chembel"
    assert resolve_dataset_subpath(tmp_path, "faers").name == "faeres"
    assert resolve_dataset_subpath(tmp_path, "pharmgkb").name == "pharmagkb"
    assert resolve_dataset_subpath(tmp_path, "twosides").name in {"twosides", "nsides"}
    assert resolve_dataset_subpath(tmp_path, "pubchem").name == "pubchem"
    assert resolve_dataset_subpath(tmp_path, "ddinter").name == "ddinter"
    assert resolve_dataset_subpath(tmp_path, "openfda").name == "openfda"

    # File lookup through alias
    chem_file = resolve_dataset_subpath(tmp_path, "chembl", "structures.smiles.tsv")
    assert chem_file.is_file()

    edges_file = resolve_dataset_subpath(tmp_path, "twosides", "drug_drug_edges.csv")
    assert edges_file.is_file()


def test_expanded_pharmgkb_bridge_with_folder_paths(tmp_path: Path):
    import pandas as pd
    from src.data_prep.expanded_pharmgkb_bridge import build_expanded_pharmgkb_profiles

    smi_a = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    smi_b = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # Caffeine

    nsides_dir = tmp_path / "nsides" / "twosides"
    nsides_dir.mkdir(parents=True)
    pd.DataFrame({"source": [smi_a], "target": [smi_b]}).to_csv(nsides_dir / "drug_drug_edges.csv", index=False)

    pharmagkb_dir = tmp_path / "pharmagkb"
    pharmagkb_dir.mkdir()
    pd.DataFrame({
        "PharmGKB Accession Id": ["PA448443", "PA448444"],
        "Name": ["Aspirin", "Caffeine"],
        "Generic Names": ["acetylsalicylic acid", "caffeine"],
        "Trade Names": ["Bayer", "No-Doz"],
        "Brand Mixtures": ["", ""],
        "SMILES": [smi_a, smi_b],
    }).to_csv(pharmagkb_dir / "chemicals.tsv", sep="\t", index=False)

    pd.DataFrame({
        "Entity1_id": ["PA448443", "PA448444"],
        "Entity1_name": ["Aspirin", "Caffeine"],
        "Entity1_type": ["Chemical", "Chemical"],
        "Entity2_id": ["GENE1", "GENE2"],
        "Entity2_name": ["CYP2C9", "CYP1A2"],
        "Entity2_type": ["Gene", "Gene"],
    }).to_csv(pharmagkb_dir / "relationships.tsv", sep="\t", index=False)

    out_csv = tmp_path / "expanded_profiles.csv"
    df_prof, summary = build_expanded_pharmgkb_profiles(
        twosides_edges_path=tmp_path / "nsides",
        pharmgkb_chemicals_path=pharmagkb_dir,
        output_profiles_path=out_csv,
    )
    assert summary["drugs_with_gene_profiles"] == 2
    assert out_csv.is_file()

