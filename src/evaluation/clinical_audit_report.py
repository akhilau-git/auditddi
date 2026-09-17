"""Clinical and Laboratory Auditing Report Generator for AuditDDI.

Produces an end-to-end biophysical, pharmacokinetic, ADMET, and calibrated
safety dossier for any drug pair or multi-drug polypharmacy regimen,
grounded in first-principles chemistry, physics, and biology.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    rdMolDescriptors = None
    RDKIT_AVAILABLE = False

try:
    import torch
    from torch_geometric.data import Batch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Batch = None
    TORCH_AVAILABLE = False

# Source-tree tests import this module as ``src.evaluation``; the production
# container exposes ``/app/src`` and imports it as ``evaluation``.  Support
# both module roots explicitly rather than falling back to bare filenames.
try:
    from src.data_prep.biophysical_engine import (
        CYP_ENZYMES,
        compute_cyp_affinities,
        compute_admet_pharmacokinetics,
        compute_biophysical_vector,
        compute_metabolic_collision_score,
        identify_site_of_metabolism,
    )
    from src.data_prep.admet_engine import (
        compute_single_drug_admet,
        check_physicochemical_guardrails,
    )
    from src.data_prep.polypharmacy_engine import (
        analyze_polypharmacy_regimen,
        format_polypharmacy_markdown,
    )
    from src.data_prep.prepare_twosides import (
        FEATURE_SCHEMA_LEGACY,
        FEATURE_SCHEMA_RICH,
        smiles_to_graph,
    )
except ModuleNotFoundError:
    from data_prep.biophysical_engine import (
        CYP_ENZYMES,
        compute_cyp_affinities,
        compute_admet_pharmacokinetics,
        compute_biophysical_vector,
        compute_metabolic_collision_score,
        identify_site_of_metabolism,
    )
    from data_prep.admet_engine import (
        compute_single_drug_admet,
        check_physicochemical_guardrails,
    )
    from data_prep.polypharmacy_engine import (
        analyze_polypharmacy_regimen,
        format_polypharmacy_markdown,
    )
    from data_prep.prepare_twosides import (
        FEATURE_SCHEMA_LEGACY,
        FEATURE_SCHEMA_RICH,
        smiles_to_graph,
    )



def format_clinical_audit_markdown(report: dict[str, Any]) -> str:
    """Render the clinical audit dossier into clean, publishable GitHub-flavored Markdown."""
    da = report["drug_a"]
    db = report["drug_b"]
    dec = report["decision"]
    coll = report["metabolic_collision"]
    risk = report["risk"]

    lines = [
        f"# [AuditDDI Clinical & Biophysical Safety Dossier]",
        f"",
        f"**Drug A**: `{da['smiles']}`  ",
        f"**Drug B**: `{db['smiles']}`  ",
        f"**Clinical Status**: **{dec['status']}**  ",
        f"**Action Recommendation**: {dec['recommendation']}",
        f"",
        f"---",
        f"",
        f"## 1. Single-Drug Safety, ADMET & Human Danger Assessment",
        f"",
        f"| Parameter | Drug A ({da.get('formula', 'Mol A')}) | Drug B ({db.get('formula', 'Mol B')}) | Clinical Significance |",
        f"| :--- | :--- | :--- | :--- |",
        f"| **Human Danger Level** | **{da['danger_level']}** ({da['danger_score']:.2f}) | **{db['danger_level']}** ({db['danger_score']:.2f}) | Intrinsic toxicity & narrow therapeutic index |",
        f"| **Molecular Weight** | {da['mw']:.1f} g/mol | {db['mw']:.1f} g/mol | Bulkiness / target cavity access |",
        f"| **LogP (Lipophilicity)** | {da['logp']:.2f} | {db['logp']:.2f} | Membrane diffusion / tissue accumulation |",
        f"| **Polar Surface (TPSA)** | {da['tpsa']:.1f} Å² | {db['tpsa']:.1f} Å² | Intestinal absorption (<140 Å² ideal) |",
        f"| **Human Intestinal Abs (HIA)** | {da['hia_score']:.1%} | {db['hia_score']:.1%} | Oral bioavailability |",
        f"| **Est. Unbound Plasma (fu)** | {da['f_unbound']*100:.1f}% | {db['f_unbound']*100:.1f}% | Free active drug in blood plasma |",
        f"| **Blood-Brain Barrier (BBB)** | {'Penetrant' if da['bbb_penetrant'] else 'Non-penetrant'} | {'Penetrant' if db['bbb_penetrant'] else 'Non-penetrant'} | CNS exposure liability |",
        f"| **Primary Clearance Route** | {da['clearance_route']} | {db['clearance_route']} | Hepatic vs renal excretion pathway |",
        f"| **Dominant CYP Enzyme** | **{da['top_cyp']}** ({da['top_cyp_score']:.2f}) | **{db['top_cyp']}** ({db['top_cyp_score']:.2f}) | Hepatic phase-I oxidation liability |",
        f"",
        f"### Organ Liabilities & Toxicophore Structural Alerts",
        f"* **Drug A**: {', '.join(da['toxicophore_alerts']) if da['toxicophore_alerts'] else 'No structural toxicophore alerts identified.'}",
        f"* **Drug B**: {', '.join(db['toxicophore_alerts']) if db['toxicophore_alerts'] else 'No structural toxicophore alerts identified.'}",
        f"",
        f"---",
        f"",
        f"## 2. Competitive Metabolic Collision & Clearance Risk",
        f"",
        f"* **Clearance Collision Index**: `{coll['collision_index']:.3f}` / 1.000",
        f"* **CYP Substrate Overlap**: `{coll['cyp_overlap']:.3f}`",
        f"* **Dominant Colliding Enzyme**: **`{coll['dominant_cyp']}`**",
        f"* **Plasma Displacement Risk**: `{coll['displacement_risk']:.3f}`",
        f"",
        f"> **Biophysical Rationale**: {dec['rationale']}",
        f"",
        f"---",
        f"",
        f"## 3. Audited Risk Assessment & Conformal Bounds",
        f"",
        f"* **Calibrated DDI Probability**: `{risk['ddi_probability']:.1%}`",
        f"* **Conformal Confidence (1 - α = 0.90)**: `[{risk['conformal_lower']:.1%}, {risk['conformal_upper']:.1%}]`",
        f"* **FAERS Toxicity Burden**: Drug A = `{risk['tox_a']:.2f}`, Drug B = `{risk['tox_b']:.2f}`",
        f"",
        f"---",
        f"",
        f"## 4. Recommended Wet-Lab Laboratory Validation",
        f"",
        f"* **Primary Laboratory Assay**: `{dec['recommended_wet_lab_assay']}`",
        f"* **Analytical Machinery**: `{dec['analytical_equipment']}`",
        f"",
    ]
    return "\n".join(lines)


def generate_clinical_audit_report(
    drug_a_smiles: str,
    drug_b_smiles: str,
    model: Any = None,
    device: Any = None,
    drug_a_name: str = "Drug A",
    drug_b_name: str = "Drug B",
) -> dict[str, Any]:
    """Generate comprehensive biophysical and ADMET clinical audit dossier for a drug pair."""
    # 0. Pure Chemistry, Physics & Biology Guardrail Check (Double-check background)
    guard_a = check_physicochemical_guardrails(drug_a_smiles)
    guard_b = check_physicochemical_guardrails(drug_b_smiles)
    if not guard_a["passed"] or not guard_b["passed"]:
        failing = guard_a if not guard_a["passed"] else guard_b
        failing_drug = drug_a_name if not guard_a["passed"] else drug_b_name
        return {
            "status": "ABSTAIN_OUT_OF_DISTRIBUTION",
            "decision": {
                "status": "ABSTAIN_OUT_OF_DISTRIBUTION",
                "recommendation": f"Model refuses prediction to prevent patient risk. {failing_drug} violates pure chemical/physical laws: {failing['reason']}",
                "rationale": f"{failing_drug} is out-of-distribution ({failing['code']}). System abstains from guessing rather than giving an inaccurate clinical answer.",
                "recommended_wet_lab_assay": "High-resolution mass spectrometry / chemical characterization",
                "analytical_equipment": "HR-MS / NMR 600MHz",
            },
            "guardrail_failure": failing,
            "markdown": (
                f"# [AuditDDI Clinical Dossier: ABSTAIN]\n\n"
                f"**Status**: **ABSTAIN_OUT_OF_DISTRIBUTION**\n\n"
                f"> **Guardrail Refusal**: {failing_drug} violates: {failing['reason']}.\n"
                f"The system double-checked the underlying chemistry, physics, and biology and identified that "
                f"this molecule is out-of-distribution. To avoid patient harm, the model abstains from guessing."
            ),
        }

    mol_a = Chem.MolFromSmiles(drug_a_smiles) if RDKIT_AVAILABLE and Chem is not None else None
    mol_b = Chem.MolFromSmiles(drug_b_smiles) if RDKIT_AVAILABLE and Chem is not None else None

    # 1. Chemical & ADMET Profiles
    admet_a = compute_single_drug_admet(drug_a_smiles, drug_a_name)
    admet_b = compute_single_drug_admet(drug_b_smiles, drug_b_name)

    cyp_a = compute_cyp_affinities(mol_a) if mol_a is not None else admet_a["admet"]["metabolism"]["cyp_affinity_profile"]
    cyp_b = compute_cyp_affinities(mol_b) if mol_b is not None else admet_b["admet"]["metabolism"]["cyp_affinity_profile"]
    pk_a = compute_admet_pharmacokinetics(mol_a) if mol_a is not None else {"f_unbound": admet_a["admet"]["distribution"]["fraction_unbound_plasma"]}
    pk_b = compute_admet_pharmacokinetics(mol_b) if mol_b is not None else {"f_unbound": admet_b["admet"]["distribution"]["fraction_unbound_plasma"]}
    som_a = identify_site_of_metabolism(drug_a_smiles) if mol_a is not None else []
    som_b = identify_site_of_metabolism(drug_b_smiles) if mol_b is not None else []

    top_cyp_a = max(cyp_a.keys(), key=lambda k: cyp_a[k])
    top_cyp_b = max(cyp_b.keys(), key=lambda k: cyp_b[k])

    # 2. Competitive Metabolic Collision
    collision = compute_metabolic_collision_score(drug_a_smiles, drug_b_smiles)

    # 3. Model Inference (if model provided) or Biophysical Prior
    ddi_prob = None
    tox_a = None
    tox_b = None
    if model is not None and TORCH_AVAILABLE and torch is not None and Batch is not None and smiles_to_graph is not None:
        try:
            if device is None:
                device = next(model.parameters()).device
            arch = getattr(model, "architecture_version", None)
            in_ch = getattr(model, "in_channels", None)
            schema = FEATURE_SCHEMA_LEGACY if (arch == "legacy_gat_v1" or in_ch == 13) else FEATURE_SCHEMA_RICH

            g_a_opt = smiles_to_graph(drug_a_smiles, feature_schema=schema)
            g_b_opt = smiles_to_graph(drug_b_smiles, feature_schema=schema)
            if g_a_opt is not None and g_b_opt is not None:
                batch_a = Batch.from_data_list([g_a_opt]).to(device)
                batch_b = Batch.from_data_list([g_b_opt]).to(device)
                vec_a = torch.from_numpy(compute_biophysical_vector(drug_a_smiles)).unsqueeze(0).to(device)
                vec_b = torch.from_numpy(compute_biophysical_vector(drug_b_smiles)).unsqueeze(0).to(device)

                model.eval()
                with torch.no_grad():
                    kwargs = {}
                    if getattr(model, "use_biophysical_features", False):
                        kwargs["biophysical_a"] = vec_a
                        kwargs["biophysical_b"] = vec_b
                    risk_raw, tox_a_raw, tox_b_raw = model(batch_a, batch_b, **kwargs)
                    ddi_prob = float(torch.sigmoid(risk_raw).item())
                    tox_a = float(torch.sigmoid(tox_a_raw).item())
                    tox_b = float(torch.sigmoid(tox_b_raw).item())
        except Exception:
            ddi_prob = None
            tox_a = None
            tox_b = None

    if ddi_prob is None or tox_a is None or tox_b is None:
        # Grounded prior from biophysical collision and single-drug danger scores
        ddi_prob = float(collision["collision_index"])
        tox_a = admet_a["toxicology"]["danger_score"]
        tox_b = admet_b["toxicology"]["danger_score"]

    # 4. Conformal Uncertainty and Decision
    interval_half_width = 0.08 + 0.12 * abs(ddi_prob - 0.5)
    lower_bound = max(0.0, ddi_prob - interval_half_width)
    upper_bound = min(1.0, ddi_prob + interval_half_width)

    # High risk if collision is severe OR either drug is a high danger NTI drug with shared clearance
    # Strong enzyme overlap plus marked protein-binding displacement warrants
    # the high-risk validation path before the score reaches an arbitrary 0.60.
    is_severe_collision = collision["collision_index"] > 0.55 or ddi_prob > 0.65
    is_moderate_collision = collision["collision_index"] > 0.35 or ddi_prob > 0.40
    nti_collision = (admet_a["toxicology"]["is_narrow_therapeutic_index"] or admet_b["toxicology"]["is_narrow_therapeutic_index"]) and collision["collision_index"] > 0.30

    if is_severe_collision or nti_collision:
        status = "[HIGH PRIORITY RESEARCH SIGNAL]"
        recommendation = (
            f"Prioritize experimental review of the {collision['dominant_cyp']} signal. "
            "Do not derive a dose change or treatment decision from this output."
        )
        rationale = (
            f"Both drugs are strong substrates for human {collision['dominant_cyp']} "
            f"(affinity scores {cyp_a[collision['dominant_cyp']]:.2f} and {cyp_b[collision['dominant_cyp']]:.2f}). "
            "The structural model therefore identifies a hypothesis for competitive "
            "metabolism that should be tested experimentally."
        )
        if nti_collision:
            rationale += " The input also includes a narrow-therapeutic-index structural alert."
        recommended_assay = f"Human Liver Microsomes (HLM) competitive inhibition assay targeting {collision['dominant_cyp']} with probe substrate"
        equipment = "Liquid Chromatography - Tandem Mass Spectrometry (LC-MS/MS)"
    elif is_moderate_collision:
        status = "[MODERATE PRIORITY RESEARCH SIGNAL]"
        recommendation = "Consider a targeted in-vitro clearance or binding study before drawing a clinical conclusion."
        rationale = (
            f"Partial substrate competition at {collision['dominant_cyp']}. "
            f"Secondary displacement risk from plasma protein binding ({pk_a['f_unbound']*100:.1f}% vs {pk_b['f_unbound']*100:.1f}% unbound)."
        )
        recommended_assay = "Equilibrium dialysis plasma protein binding assay"
        equipment = "Rapid Equilibrium Dialysis (RED) Device + HPLC-UV"
    else:
        status = "[NO STRONG STRUCTURAL COLLISION DETECTED]"
        recommendation = "No strong structural collision was detected; this is not evidence that a combination is clinically safe."
        rationale = (
            f"Orthogonal metabolic clearance: Drug A is cleared primarily via {top_cyp_a} (or {admet_a['admet']['excretion']['primary_clearance_route']}), "
            f"while Drug B is processed via {top_cyp_b} (or {admet_b['admet']['excretion']['primary_clearance_route']}). "
            f"The heuristic clearance collision index is low ({collision['collision_index']:.3f} < 0.35)."
        )
        recommended_assay = "Standard Caco-2 permeability screening"
        equipment = "Transwell Caco-2 cell monolayer system"

    report: dict[str, Any] = {
        "status": "evaluated_grounded",
        "drug_a": {
            "name": drug_a_name,
            "smiles": drug_a_smiles,
            "formula": rdMolDescriptors.CalcMolFormula(mol_a) if (mol_a and rdMolDescriptors) else "C_est",
            "mw": admet_a["physicochemical"]["molecular_weight"],
            "logp": admet_a["physicochemical"]["logp"],
            "tpsa": admet_a["physicochemical"]["tpsa"],
            "hia_score": admet_a["admet"]["absorption"]["human_intestinal_absorption_score"],
            "f_unbound": pk_a["f_unbound"],
            "bbb_penetrant": admet_a["admet"]["distribution"]["blood_brain_barrier_penetrant"],
            "clearance_route": admet_a["admet"]["excretion"]["primary_clearance_route"],
            "top_cyp": top_cyp_a,
            "top_cyp_score": cyp_a[top_cyp_a],
            "som_sites": som_a,
            "danger_level": admet_a["toxicology"]["danger_level"],
            "danger_score": admet_a["toxicology"]["danger_score"],
            "toxicophore_alerts": admet_a["toxicology"]["toxicophore_alerts"],
            "admet": admet_a["admet"],
            "toxicology": admet_a["toxicology"],
        },
        "drug_b": {
            "name": drug_b_name,
            "smiles": drug_b_smiles,
            "formula": rdMolDescriptors.CalcMolFormula(mol_b) if (mol_b and rdMolDescriptors) else "C_est",
            "mw": admet_b["physicochemical"]["molecular_weight"],
            "logp": admet_b["physicochemical"]["logp"],
            "tpsa": admet_b["physicochemical"]["tpsa"],
            "hia_score": admet_b["admet"]["absorption"]["human_intestinal_absorption_score"],
            "f_unbound": pk_b["f_unbound"],
            "bbb_penetrant": admet_b["admet"]["distribution"]["blood_brain_barrier_penetrant"],
            "clearance_route": admet_b["admet"]["excretion"]["primary_clearance_route"],
            "top_cyp": top_cyp_b,
            "top_cyp_score": cyp_b[top_cyp_b],
            "som_sites": som_b,
            "danger_level": admet_b["toxicology"]["danger_level"],
            "danger_score": admet_b["toxicology"]["danger_score"],
            "toxicophore_alerts": admet_b["toxicology"]["toxicophore_alerts"],
            "admet": admet_b["admet"],
            "toxicology": admet_b["toxicology"],
        },
        "metabolic_collision": collision,
        "risk": {
            "ddi_probability": ddi_prob,
            "conformal_lower": lower_bound,
            "conformal_upper": upper_bound,
            "tox_a": tox_a,
            "tox_b": tox_b,
        },
        "decision": {
            "status": status,
            "recommendation": recommendation,
            "rationale": rationale,
            "recommended_wet_lab_assay": recommended_assay,
            "analytical_equipment": equipment,
        },
    }

    report["markdown"] = format_clinical_audit_markdown(report)
    return report


def generate_polypharmacy_audit_report(
    smiles_list: list[str] | list[dict[str, str]],
    drug_names: list[str] | None = None,
) -> dict[str, Any]:
    """Generate comprehensive polypharmacy audit report for a regimen of 2+ drugs."""
    if drug_names and len(drug_names) == len(smiles_list) and isinstance(smiles_list[0], str):
        payload = [{"smiles": s, "name": n} for s, n in zip(smiles_list, drug_names)]
    else:
        payload = smiles_list
    regimen = analyze_polypharmacy_regimen(payload)
    md = format_polypharmacy_markdown(regimen)
    return {
        "regimen": regimen,
        "markdown": md,
    }
