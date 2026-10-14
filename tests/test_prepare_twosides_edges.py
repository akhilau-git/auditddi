import gzip
from pathlib import Path

import pandas as pd

from scripts.prepare_twosides_edges import Lookup, lookup_pubchem, prepare_edges


class Response:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


def test_lookup_falls_back_to_rxnorm_identifier():
    calls = []

    def get(url, timeout):
        calls.append(url)
        if "/compound/name/" in url:
            return Response(404)
        if "/xref/RN/123/cids/TXT" in url:
            return Response(200, "456\n")
        return Response(200, "CCO\n")

    result = lookup_pubchem("Unknown product", 123, request_get=get, sleep=lambda _: None)

    assert result.smiles == "CCO"
    assert result.status == "matched_by_rxnorm"
    assert len(calls) == 3


def test_prepare_edges_writes_audit_report_and_keeps_unresolved_out(tmp_path: Path, monkeypatch):
    raw = tmp_path / "TWOSIDES.csv.gz"
    with gzip.open(raw, "wt", newline="") as handle:
        pd.DataFrame(
            {
                "drug_1_concept_name": ["Known", "Unknown"],
                "drug_2_concept_name": ["Known", "Known"],
                "drug_1_rxnorm_id": [1, 2],
                "drug_2_rxnorm_id": [1, 1],
                "condition_concept_name": ["Pain", "Pain"],
            }
        ).to_csv(handle, index=False)

    monkeypatch.setattr(
        "scripts.prepare_twosides_edges.lookup_pubchem",
        lambda name, rxnorm: Lookup(
            "CCO" if name == "Known" else None,
            "matched" if name == "Known" else "not_found",
            1,
            "",
        ),
    )
    output = tmp_path / "edges.csv"
    report = tmp_path / "unresolved.json"
    summary = prepare_edges(raw, output, report_path=report, delay=0)

    assert summary == {
        "unique_names": 2,
        "unresolved_names": 1,
        "rows_written": 1,
        "rows_skipped": 1,
    }
    assert len(pd.read_csv(output)) == 1
    assert "Unknown" in report.read_text(encoding="utf-8")
