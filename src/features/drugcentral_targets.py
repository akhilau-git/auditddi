"""Connect DrugCentral to the 645 TWOSIDES drugs (targets + a name bridge).

    python -m src.features.drugcentral_targets \
        --table /content/drive/MyDrive/auditddi-data/event_table_v1 \
        --drugcentral /content/drive/MyDrive/auditddi-data/drugcentral \
        --chembl-targets /content/drive/MyDrive/auditddi-data/p1_targets/drug_targets_chembl_mechanism.csv \
        --out /content/drive/MyDrive/auditddi-data/p1_targets

Inputs (as downloaded):
  structures.smiles.tsv          SMILES InChI InChIKey ID INN CAS_RN
  drug_target_interaction.tsv.gz DRUG_NAME STRUCT_ID TARGET_NAME TARGET_CLASS ACCESSION GENE ... MOA ... ACTION_TYPE ORGANISM
Outputs:
  drug_targets_drugcentral.csv   drug_idx, uniprot, gene, organism, target_class, moa, action_type, source
  drug_targets_union.csv         ChEMBL-mechanism + DrugCentral targets (same columns subset: drug_idx, uniprot, organism, source)
  drug_names_drugcentral.csv     drug_idx, drugcentral_id, inn, cas_rn   <- the NAME BRIDGE for DDInter / openFDA / RxNorm
  drugcentral_match_report.json  coverage numbers

Matching rule: the connectivity layer of the InChIKey (first 14 characters) of the LARGEST fragment,
so salts and stereoisomers still match, but different skeletons never do.  Unmatched drugs stay
unmatched; nothing is guessed.  ACCESSION fields that list several accessions (protein complexes)
are split into one row per accession.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


def rdkit_ik14(smiles: str) -> Optional[str]:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import inchi  # type: ignore

    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    big = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    key = inchi.MolToInchiKey(big)
    return key[:14] if key else None


def split_accessions(x) -> list:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    return [a for a in re.split(r"[|,;\s]+", str(x).strip()) if a and a.lower() != "nan"]


def match_structures(drugs: pd.DataFrame, structures: pd.DataFrame, ik_fn: Callable[[str], Optional[str]] = rdkit_ik14,
                     smiles_col: str = "SMILES", id_col: str = "ID") -> pd.DataFrame:
    """One row per matched (drug_idx, DrugCentral ID).  A skeleton shared by several DrugCentral
    structures keeps all of them; a TWOSIDES drug never gets a guessed match."""
    d = drugs[["drug_idx", "smiles"]].copy()
    d["ik"] = d["smiles"].map(ik_fn)
    s = structures.copy()
    s["ik"] = s[smiles_col].map(ik_fn)
    s = s.dropna(subset=["ik"])
    m = d.dropna(subset=["ik"]).merge(s, on="ik", how="inner")
    return m.rename(columns={id_col: "drugcentral_id"})


def build_targets(matched: pd.DataFrame, targets: pd.DataFrame, moa_only: bool = False) -> pd.DataFrame:
    t = targets.copy()
    if moa_only and "MOA" in t:
        t = t[t["MOA"].astype(str).isin(["1", "1.0", "True", "true"])]
    t = t.merge(matched[["drug_idx", "drugcentral_id"]], left_on="STRUCT_ID", right_on="drugcentral_id", how="inner")
    rows = []
    for r in t.itertuples():
        for acc in split_accessions(getattr(r, "ACCESSION", None)):
            rows.append({"drug_idx": int(r.drug_idx), "uniprot": acc, "gene": getattr(r, "GENE", None), "organism": getattr(r, "ORGANISM", None),
                         "target_class": getattr(r, "TARGET_CLASS", None), "moa": getattr(r, "MOA", None),
                         "action_type": getattr(r, "ACTION_TYPE", None), "source": "drugcentral"})
    cols = ["drug_idx", "uniprot", "gene", "organism", "target_class", "moa", "action_type", "source"]
    out = pd.DataFrame(rows, columns=cols).drop_duplicates(["drug_idx", "uniprot"])
    return out


def union_targets(chembl: pd.DataFrame, dc: pd.DataFrame) -> pd.DataFrame:
    a = chembl[["drug_idx", "uniprot", "organism"]].assign(source="chembl_mechanism", target_class=np.nan)
    b = dc[["drug_idx", "uniprot", "organism", "target_class"]].assign(source="drugcentral")
    u = pd.concat([a, b], ignore_index=True)
    # one row per (drug, target); if both sources have it, record both and keep the DrugCentral class
    u = u.groupby(["drug_idx", "uniprot"], as_index=False).agg(
        organism=("organism", "first"), target_class=("target_class", "first"),
        source=("source", lambda s: "+".join(sorted(set(s)))))
    return u


def name_bridge(matched: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("drug_idx", "drugcentral_id", "INN", "CAS_RN") if c in matched]
    b = matched[cols].drop_duplicates("drug_idx").rename(columns={"INN": "inn", "CAS_RN": "cas_rn"})
    return b.sort_values("drug_idx").reset_index(drop=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, type=Path)
    ap.add_argument("--drugcentral", required=True, type=Path)
    ap.add_argument("--chembl-targets", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--moa-only", action="store_true", help="keep only mechanism-of-action targets (drop off-target activity rows)")
    ap.add_argument("--identity-key", action="store_true", help="TESTING ONLY: match on the SMILES string itself")
    a = ap.parse_args(list(argv) if argv is not None else None)

    drugs = pd.read_csv(a.table / "drugs.csv")
    st = pd.read_csv(a.drugcentral / "structures.smiles.tsv", sep="\t")
    tg = pd.read_csv(a.drugcentral / "drug_target_interaction.tsv.gz", sep="\t", compression="gzip")
    fn = (lambda s: str(s)) if a.identity_key else rdkit_ik14
    matched = match_structures(drugs, st, fn)
    dc = build_targets(matched, tg, a.moa_only)
    a.out.mkdir(parents=True, exist_ok=True)
    dc.to_csv(a.out / "drug_targets_drugcentral.csv", index=False)
    name_bridge(matched).to_csv(a.out / "drug_names_drugcentral.csv", index=False)
    rep = {"n_drugs": int(len(drugs)), "drugs_matched_to_drugcentral": int(matched.drug_idx.nunique()),
           "drugs_with_drugcentral_targets": int(dc.drug_idx.nunique()), "distinct_target_accessions": int(dc.uniprot.nunique()),
           "moa_only": bool(a.moa_only), "organisms_top": dc.organism.value_counts().head(5).to_dict()}
    if a.chembl_targets and Path(a.chembl_targets).exists():
        ch = pd.read_csv(a.chembl_targets)
        u = union_targets(ch, dc)
        u.to_csv(a.out / "drug_targets_union.csv", index=False)
        rep.update({"drugs_with_chembl_targets": int(ch.drug_idx.nunique()), "drugs_with_union_targets": int(u.drug_idx.nunique()),
                    "distinct_accessions_union": int(u.uniprot.nunique())})
    (a.out / "drugcentral_match_report.json").write_text(json.dumps(rep, indent=2))
    for k, v in rep.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
