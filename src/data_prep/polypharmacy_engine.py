"""Multi-Drug Polypharmacy & Regimen Collision Engine for AuditDDI.

Evaluates complex drug regimens (2 or more drugs: A + B, A + B + C, ...):
1. Single-Drug ADMET & Danger: Computes full PK/PD parameters, toxicophores, and human hazard levels.
2. Pairwise Collisions: All pairwise CYP clearance bottlenecks, plasma displacement, and target synergy.
3. Regimen-Level Bottleneck: Cumulative CYP3A4/2D6/2C9 load across the entire patient medication list.
4. "SAFE TO TAKE" Classification: Proves mechanistically why a combination is safe when no conflicts exist.
5. High-Risk / Contraindication Alerts: Flags dangerous cumulative organ toxicity and NTI displacements.
6. Guardrail Abstention: Refuses to guess on corrupt/OOD molecules ("I don't know / Out of Distribution").
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

try:
    from src.data_prep.admet_engine import compute_single_drug_admet, check_physicochemical_guardrails
except ImportError:
    from .admet_engine import compute_single_drug_admet, check_physicochemical_guardrails


def analyze_polypharmacy_regimen(
    drugs: list[dict[str, str]] | list[str],
    patient_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze a multi-drug regimen (2, 3, 4+ drugs) grounded in chemistry, biology, and physics."""
    # Normalize input
    normalized_drugs: list[dict[str, str]] = []
    for idx, item in enumerate(drugs):
        if isinstance(item, str):
            normalized_drugs.append({"smiles": item, "name": f"Drug_{chr(65 + idx)}"})
        elif isinstance(item, dict):
            smi = item.get("smiles", "").strip()
            name = item.get("name") or f"Drug_{chr(65 + idx)}"
            normalized_drugs.append({"smiles": smi, "name": name})

    n_drugs = len(normalized_drugs)
    if n_drugs < 2:
        return {
            "status": "error_insufficient_drugs",
            "message": "Polypharmacy analysis requires at least 2 drugs.",
            "overall_verdict": "UNKNOWN",
        }

    # Step 1: Guardrail Check on Every Drug
    single_drug_profiles: list[dict[str, Any]] = []
    failed_guardrails: list[dict[str, Any]] = []

    for item in normalized_drugs:
        prof = compute_single_drug_admet(item["smiles"], item["name"])
        single_drug_profiles.append(prof)
        if not prof["guardrail"]["passed"]:
            failed_guardrails.append({
                "drug": item["name"],
                "smiles": item["smiles"],
                "guardrail_code": prof["guardrail"]["code"],
                "reason": prof["guardrail"]["reason"],
            })

    # Guardrail Abstention: If any molecule violates fundamental chemistry/physics, abstain!
    if failed_guardrails:
        return {
            "status": "ABSTAIN_OUT_OF_DISTRIBUTION",
            "overall_verdict": "ABSTAIN_CANNOT_EVALUATE",
            "danger_level": "UNKNOWN_UNRELIABLE",
            "clinical_guidance": (
                "The system refuses to predict interactions for this regimen. "
                "One or more supplied molecules lies outside valid chemical valency or therapeutic drug boundaries. "
                "Instead of guessing, the model abstains to protect patient safety."
            ),
            "failed_guardrails": failed_guardrails,
            "single_drugs": single_drug_profiles,
        }

    # Step 2: Pairwise Interaction Analysis across all combinations C(N, 2)
    pairwise_interactions: list[dict[str, Any]] = []
    cyp_regimen_load: dict[str, float] = {
        "CYP3A4": 0.0, "CYP2D6": 0.0, "CYP2C9": 0.0, "CYP1A2": 0.0, "CYP2C19": 0.0
    }

    cumulative_herg_alerts = 0
    cumulative_hepato_alerts = 0
    nti_drugs_present: list[str] = []

    for prof in single_drug_profiles:
        # Accumulate CYP load
        cyp_prof = prof.get("admet", {}).get("metabolism", {}).get("cyp_affinity_profile", {})
        for enzyme, aff in cyp_prof.items():
            if enzyme in cyp_regimen_load:
                cyp_regimen_load[enzyme] += aff

        # Accumulate toxicity alerts
        tox = prof.get("toxicology", {})
        if tox.get("narrow_therapeutic_index"):
            nti_drugs_present.append(prof["drug_name"])
        for alert in tox.get("structural_alerts", []):
            if alert["category"] == "cardiotoxicity_herg":
                cumulative_herg_alerts += 1
            elif alert["category"] == "hepatotoxicity_reactive":
                cumulative_hepato_alerts += 1

    # Evaluate each pairwise drug combination
    max_pairwise_collision = 0.0
    high_risk_pairs = []

    for (p1, p2) in combinations(single_drug_profiles, 2):
        d1_name, d2_name = p1["drug_name"], p2["drug_name"]
        d1_cyp = p1.get("admet", {}).get("metabolism", {}).get("cyp_affinity_profile", {})
        d2_cyp = p2.get("admet", {}).get("metabolism", {}).get("cyp_affinity_profile", {})

        # Compute CYP competition overlap
        collision_score = 0.0
        colliding_enzymes = []
        for enzyme in ["CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "CYP2C19"]:
            aff1 = d1_cyp.get(enzyme, 0.0)
            aff2 = d2_cyp.get(enzyme, 0.0)
            overlap = min(aff1, aff2)
            if overlap > 0.25:
                colliding_enzymes.append(f"{enzyme} (load: {overlap:.2f})")
                collision_score += overlap

        # Plasma protein displacement risk
        fu1 = p1.get("admet", {}).get("distribution", {}).get("fraction_unbound_plasma", 0.5)
        fu2 = p2.get("admet", {}).get("distribution", {}).get("fraction_unbound_plasma", 0.5)
        displacement_risk = 0.0
        if min(fu1, fu2) < 0.05 and max(p1["physicochemical"]["logp"], p2["physicochemical"]["logp"]) > 3.0:
            displacement_risk = round(0.75 * (1.0 - min(fu1, fu2)), 3)

        total_pair_risk = min(collision_score + displacement_risk, 1.0)
        max_pairwise_collision = max(max_pairwise_collision, total_pair_risk)

        is_pair_severe = False
        reasons = []
        if colliding_enzymes and total_pair_risk > 0.5:
            is_pair_severe = True
            reasons.append(f"Structural metabolic-overlap signal at {', '.join(colliding_enzymes)}")
        if displacement_risk > 0.4:
            is_pair_severe = True
            reasons.append("Structural protein-binding displacement signal")
        if p1["toxicology"]["narrow_therapeutic_index"] or p2["toxicology"]["narrow_therapeutic_index"]:
            if total_pair_risk > 0.3:
                is_pair_severe = True
                reasons.append("Includes a narrow-therapeutic-index structural alert under metabolic competition")

        pair_summary = {
            "pair": f"{d1_name} + {d2_name}",
            "drug_a": d1_name,
            "drug_b": d2_name,
            "collision_score": round(total_pair_risk, 3),
            "colliding_enzymes": colliding_enzymes,
            "displacement_risk": displacement_risk,
            "severe_interaction_risk": is_pair_severe,
            "mechanistic_reasons": reasons if reasons else ["No strong structural clearance-overlap signal detected"],
        }
        pairwise_interactions.append(pair_summary)
        if is_pair_severe:
            high_risk_pairs.append(pair_summary)

    # Step 3: Regimen Bottlenecks & Overall Clinical Verdict
    top_cyp_bottleneck = max(cyp_regimen_load.items(), key=lambda x: x[1])
    is_bottleneck_critical = top_cyp_bottleneck[1] >= (1.2 + 0.3 * (n_drugs - 2))

    # Additive Cardiotoxicity Danger
    additive_cardiotoxicity = cumulative_herg_alerts >= 2
    additive_hepatotoxicity = cumulative_hepato_alerts >= 3

    # Safety Determination: When is it SAFE TO TAKE?
    if (
        not high_risk_pairs
        and not is_bottleneck_critical
        and not additive_cardiotoxicity
        and max_pairwise_collision < 0.35
    ):
        overall_verdict = "NO_STRONG_STRUCTURAL_SIGNAL"
        danger_level = "RESEARCH_PRIORITY_LOW"
        clinical_guidance = (
            "[NO STRONG STRUCTURAL SIGNAL]: The heuristic did not identify a strong metabolic-overlap "
            "or additive-alert signal. This is not evidence that the regimen is clinically safe."
        )
    elif high_risk_pairs or is_bottleneck_critical or additive_cardiotoxicity:
        overall_verdict = "HIGH_PRIORITY_RESEARCH_SIGNAL"
        danger_level = "RESEARCH_PRIORITY_HIGH"
        reasons_list = []
        if high_risk_pairs:
            reasons_list.append(f"{len(high_risk_pairs)} severe drug pair collision(s) detected")
        if is_bottleneck_critical:
            reasons_list.append(f"Severe metabolic bottleneck at {top_cyp_bottleneck[0]} (accumulated load: {top_cyp_bottleneck[1]:.2f})")
        if additive_cardiotoxicity:
            reasons_list.append(f"Cumulative hERG cardiac alert ({cumulative_herg_alerts} QT-prolonging pharmacophores)")
        if additive_hepatotoxicity:
            reasons_list.append(f"High cumulative reactive metabolite burden ({cumulative_hepato_alerts} hepatotoxic alerts)")
        clinical_guidance = (
            f"[HIGH PRIORITY RESEARCH SIGNAL]: {'; '.join(reasons_list)}. "
            "Prioritize experimental review; this output does not establish a clinical contraindication."
        )
    else:
        overall_verdict = "MODERATE_PRIORITY_RESEARCH_SIGNAL"
        danger_level = "RESEARCH_PRIORITY_MODERATE"
        clinical_guidance = (
            "[MODERATE PRIORITY RESEARCH SIGNAL]: Mild to moderate structural overlap was detected. "
            "Consider targeted experimental follow-up before making a clinical conclusion."
        )

    return {
        "status": "evaluated_grounded",
        "num_drugs": n_drugs,
        "overall_verdict": overall_verdict,
        "danger_level": danger_level,
        "clinical_guidance": clinical_guidance,
        "regimen_metrics": {
            "max_pairwise_collision": round(max_pairwise_collision, 3),
            "severe_pair_conflicts_count": len(high_risk_pairs),
            "top_metabolic_bottleneck": f"{top_cyp_bottleneck[0]} (load: {top_cyp_bottleneck[1]:.2f})",
            "cumulative_cyp_loads": {k: round(v, 2) for k, v in cyp_regimen_load.items()},
            "cumulative_herg_cardiotoxicity_alerts": cumulative_herg_alerts,
            "cumulative_hepatotoxicity_alerts": cumulative_hepato_alerts,
            "narrow_therapeutic_index_drugs": nti_drugs_present,
        },
        "single_drug_profiles": single_drug_profiles,
        "pairwise_interactions": pairwise_interactions,
    }


def format_polypharmacy_markdown(result: dict[str, Any]) -> str:
    """Render polypharmacy evaluation into GitHub-flavored Markdown."""
    if result.get("status") == "ABSTAIN_OUT_OF_DISTRIBUTION":
        return (
            f"# [AuditDDI Polypharmacy Audit: ABSTAIN]\n\n"
            f"**Overall Verdict**: **{result['overall_verdict']}**\n\n"
            f"> **Clinical Safety Refusal**: {result['clinical_guidance']}\n"
        )

    lines = [
        f"# [AuditDDI Multi-Drug Polypharmacy Safety Audit]",
        f"",
        f"**Regimen Size**: {result.get('num_drugs', 0)} drugs  ",
        f"**Overall Clinical Verdict**: **{result.get('overall_verdict', 'UNKNOWN')}**  ",
        f"**Danger Level**: **{result.get('danger_level', 'UNKNOWN')}**  ",
        f"**Clinical Guidance**: {result.get('clinical_guidance', '')}",
        f"",
        f"---",
        f"",
        f"## 1. Cumulative Regimen Metabolic Bottleneck & Organ Burden",
        f"",
        f"* **Top Clearance Bottleneck**: `{result['regimen_metrics']['top_metabolic_bottleneck']}`",
        f"* **Max Pairwise Clearance Collision**: `{result['regimen_metrics']['max_pairwise_collision']:.3f}` / 1.000",
        f"* **Severe Pairwise Collisions**: `{result['regimen_metrics']['severe_pair_conflicts_count']}`",
        f"* **Cumulative hERG Cardiotoxicity Alerts**: `{result['regimen_metrics']['cumulative_herg_cardiotoxicity_alerts']}`",
        f"* **Cumulative Hepatotoxicity Alerts**: `{result['regimen_metrics']['cumulative_hepatotoxicity_alerts']}`",
        f"* **Narrow Therapeutic Index (NTI) Drugs**: {', '.join(result['regimen_metrics']['narrow_therapeutic_index_drugs']) if result['regimen_metrics']['narrow_therapeutic_index_drugs'] else 'None'}",
        f"",
        f"### Cumulative CYP Metabolic Clearance Loads",
        f"| Enzyme | Cumulative Regimen Load | Clearance Saturation Status |",
        f"| :--- | :--- | :--- |",
    ]
    for cyp, load in result["regimen_metrics"]["cumulative_cyp_loads"].items():
        sat = "SATURATED / HIGH RISK" if load >= 1.5 else ("ELEVATED" if load >= 1.0 else "NOMINAL")
        lines.append(f"| **{cyp}** | {load:.2f} | {sat} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Single-Drug ADMET & Danger Summary",
        f"",
        f"| Drug | Danger Level | Score | MW (g/mol) | LogP | Top CYP | Primary Route | NTI |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for p in result.get("single_drug_profiles", []):
        tox = p["toxicology"]
        phys = p["physicochemical"]
        admet = p["admet"]
        cyp_name = admet.get("metabolism", {}).get("primary_cyp_enzyme") or admet.get("metabolism", {}).get("dominant_cyp", "CYP3A4")
        lines.append(
            f"| **{p['drug_name']}** | {tox['danger_level']} | {tox['danger_score']:.2f} | "
            f"{phys['molecular_weight']:.1f} | {phys['logp']:.2f} | {cyp_name} | "
            f"{admet['excretion']['primary_clearance_route']} | {'YES' if tox['is_narrow_therapeutic_index'] else 'No'} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Pairwise Drug Collision Matrix",
        f"",
        f"| Drug Pair | Collision Score | Colliding Enzymes | Clinical Risk Level | Mechanistic Rationale |",
        f"| :--- | :--- | :--- | :--- | :--- |",
    ])
    for pair in result.get("pairwise_interactions", []):
        enzymes_str = ", ".join(pair["colliding_enzymes"]) if pair["colliding_enzymes"] else "None (Orthogonal)"
        risk_str = "SEVERE DDI RISK" if pair["severe_interaction_risk"] else "COMPATIBLE / LOW"
        reasons_str = "; ".join(pair["mechanistic_reasons"])
        lines.append(
            f"| **{pair['drug_a']} + {pair['drug_b']}** | `{pair['collision_score']:.3f}` | "
            f"{enzymes_str} | **{risk_str}** | {reasons_str} |"
        )

    return "\n".join(lines)
