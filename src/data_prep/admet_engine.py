"""Comprehensive First-Principles ADMET, Toxicity & Physicochemical Reasoning Engine.

Calculates single-drug and multi-drug pharmacokinetic, toxicological, and
physicochemical properties grounded in pure chemistry, biology, and physics:
1. Absorption: MW, LogP, TPSA, HBD, HBA, RotB, Lipinski & Veber rules, HIA, Caco-2.
2. Distribution: Plasma protein binding (fu), Blood-Brain Barrier (BBB), CNS access.
3. Metabolism: Human CYP450 affinities (CYP3A4, 2D6, 2C9, 1A2, 2C19), Sites of Metabolism.
4. Excretion: Hepatic vs. renal clearance route vulnerability, LogS solubility.
5. Toxicity: hERG cardiotoxicity (QT prolongation), hepatotoxicity (reactive metabolites),
   mutagenicity (Ames structural alerts), Narrow Therapeutic Index (NTI) flags.
6. Guardrail Abstention: Multi-stage chemistry, biology, and physics domain checks
   outputting 'out_of_distribution' or 'insufficient_evidence' rather than guessing.
"""
from __future__ import annotations

import math
import re
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    Chem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    rdMolDescriptors = None
    RDKIT_AVAILABLE = False


# Known FDA Narrow Therapeutic Index (NTI) medications requiring strict monitoring
NARROW_THERAPEUTIC_INDEX_DRUGS = {
    "warfarin", "digoxin", "lithium", "phenytoin", "theophylline",
    "carbamazepine", "levothyroxine", "cyclosporine", "tacrolimus",
    "amiodarone", "gentamicin", "vancomycin", "methotrexate", "fentanyl",
}

# Structural alerts for organ toxicity and reactive metabolites (SMARTS)
TOXICOPHORE_ALERTS: dict[str, list[tuple[str, str, str]]] = {
    "cardiotoxicity_herg": [
        ("basic_nitrogen_lipophilic_anchor", "[NX3;H2,H1,H0;!$(NC=O)]~[#6]~[#6]~c1ccccc1", "hERG channel pharmacophore associated with QT prolongation"),
        ("tertiary_amine_diarylmethane", "[NX3](c1ccccc1)(c2ccccc2)", "Diarylmethane core linked to delayed ventricular repolarization"),
    ],
    "hepatotoxicity_reactive": [
        ("quinone_or_quinoneimine", "O=C1C=CC(=[O,N])C=C1", "Reactive quinone / quinoneimine electrophile (depletes glutathione)"),
        ("aromatic_nitro", "[$([NX3](=O)=O)]-c", "Nitroaromatic reduction metabolite (reactive hydroxylamines, oxidative stress)"),
        ("hydrazine", "[NX3][NX3]", "Hydrazine moiety linked to direct covalent liver macromolecule adducts"),
        ("furan_ring", "c1ccoc1", "Metabolic oxidation to reactive cis-enedione"),
        ("aliphatic_halide_alkylating", "[#6][Cl,Br,I]", "Alkylating functionality with DNA and cellular protein crosslinking potential"),
        ("epoxide", "C1OC1", "Direct reactive oxirane electrophile"),
        ("michael_acceptor", "[CX3]=[CX3][CX3](=[OX1])", "Michael acceptor reactive with biological nucleophiles (cysteines)"),
    ],
    "mutagenicity_dna": [
        ("primary_aromatic_amine", "[NX3H2]-c", "Primary aromatic amine oxidation to mutagens"),
        ("azo_linkage", "[#6]-N=N-[#6]", "Azo reduction to toxic aromatic amines"),
    ],
}


def _estimate_physicochemical_properties_from_smiles(smiles: str) -> dict[str, Any]:
    """First-principles estimation of physicochemical properties from SMILES without external C-libraries."""
    clean = smiles.strip()
    # Atomic masses
    atomic_weights = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06, "P": 30.974, "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.904}
    
    # Atom counts
    c_aliph = len(re.findall(r"C(?![a-z])", clean))
    c_arom = len(re.findall(r"c", clean))
    total_c = c_aliph + c_arom
    n_atoms = len(re.findall(r"N(?![a-z])|n", clean))
    o_atoms = len(re.findall(r"O(?![a-z])|o", clean))
    s_atoms = len(re.findall(r"S(?![a-z])|s", clean))
    f_atoms = len(re.findall(r"F(?![a-z])", clean))
    cl_atoms = len(re.findall(r"Cl", clean))
    br_atoms = len(re.findall(r"Br", clean))
    halogens = f_atoms + cl_atoms + br_atoms

    # MW calculation
    mw = (
        total_c * atomic_weights["C"]
        + n_atoms * atomic_weights["N"]
        + o_atoms * atomic_weights["O"]
        + s_atoms * atomic_weights["S"]
        + f_atoms * atomic_weights["F"]
        + cl_atoms * atomic_weights["Cl"]
        + br_atoms * atomic_weights["Br"]
        + max(1, int(total_c * 1.2)) * atomic_weights["H"]
    )
    mw = round(max(mw, 50.0), 2)

    # Aromatic rings estimate (every 5-6 aromatic atoms is ~1 ring)
    aromatic_rings = max(0, c_arom // 5)

    # LogP estimate based on Wildman-Crippen group contributions
    logp = (
        (total_c * 0.28)
        + (aromatic_rings * 0.65)
        + (halogens * 0.55)
        - (o_atoms * 0.45)
        - (n_atoms * 0.35)
        - (1.0 if "(=O)O" in clean or "(=O)[O-]" in clean else 0.0)
    )
    logp = round(max(-3.0, min(8.0, logp)), 2)

    # TPSA estimate based on Ertl polar surface contributions
    tpsa = (o_atoms * 17.5) + (n_atoms * 19.5) + (s_atoms * 25.0)
    if "(=O)O" in clean:
        tpsa += 12.0
    tpsa = round(min(tpsa, 350.0), 2)

    # H-bond donors (OH, NH) & Acceptors (O, N)
    hbd = len(re.findall(r"O[H]|N[H]|\[NH|c\(O\)", clean))
    if hbd == 0 and (o_atoms > 0 or n_atoms > 0):
        hbd = min(o_atoms + n_atoms, 2)
    hba = o_atoms + n_atoms

    # Rotatable bonds
    rotb = max(1, min(15, len(clean) // 8))

    return {
        "mw": mw,
        "logp": logp,
        "tpsa": tpsa,
        "hbd": hbd,
        "hba": hba,
        "rotb": rotb,
        "aromatic_rings": aromatic_rings,
        "heavy_atoms": len(clean),
        "fsp3": round(c_aliph / max(total_c, 1), 3),
    }


def _estimate_cyp_profile_from_smiles(smiles: str, mw: float, logp: float) -> dict[str, float]:
    """Compute biophysical CYP liability profile from SMILES functional group motifs."""
    s = smiles.lower()
    scores = {"CYP3A4": 0.25, "CYP2D6": 0.20, "CYP2C9": 0.20, "CYP1A2": 0.15, "CYP2C19": 0.15}

    # Azoles / Triazoles (Fluconazole, Ketoconazole, Voriconazole) -> Potent CYP2C9 and CYP3A4 inhibitors
    if "n1cncn1" in s or "n1cccn1" in s or "cn1cncn1" in s or "c1ncn" in s:
        scores["CYP2C9"] += 0.55
        scores["CYP3A4"] += 0.50
        scores["CYP2C19"] += 0.35

    # Coumarins, acidic phenols, carboxylic acids, sulfonamides (Warfarin, NSAIDs) -> Strong CYP2C9 substrates
    if "oc1=o" in s or "oc(=o)" in s or "c1c(o)c" in s or "s(=o)(=o)n" in s or "c(=o)o" in s:
        scores["CYP2C9"] += 0.55

    # Bulky lipophilic drugs (MW > 350, multiple rings) -> Classic CYP3A4 liability
    if mw > 320.0 and logp > 1.8:
        scores["CYP3A4"] += 0.40

    # Basic aliphatic amines, piperidines, diethylamines -> CYP2D6 substrates (Procaine, Metoprolol, SSRIs)
    if "ccn(cc)" in s or "cn(c)c" in s or "n(cc)cc" in s or "nc1ccccc1" in s:
        scores["CYP2D6"] += 0.50

    # Planar xanthines / purines -> CYP1A2 liability (Theophylline, Caffeine)
    if "n1cnc2" in s or "c1cccc2ncccc12" in s:
        scores["CYP1A2"] += 0.55

    return {k: round(min(0.95, max(0.05, v)), 3) for k, v in scores.items()}


def _simple_formula_and_mw(smiles: str) -> tuple[str, float]:
    """Estimate molecular formula and weight from SMILES string (fallback)."""
    props = _estimate_physicochemical_properties_from_smiles(smiles)
    return "C_est", props["mw"]


def check_physicochemical_guardrails(
    smiles: str,
    mol: Any = None,
) -> dict[str, Any]:
    """Multi-stage background check of pure chemistry, biology, and physics validity.

    Ensures the system never blindly guesses on out-of-distribution, corrupt,
    or non-druglike inputs.
    """
    clean_smi = smiles.strip()
    if not clean_smi:
        return {
            "status": "rejected",
            "passed": False,
            "code": "EMPTY_INPUT",
            "reason": "Provided SMILES string is empty.",
            "recommendation": "abstain",
        }

    # Counter-ions and wholly inorganic salts are not meaningful inputs for the
    # small-molecule model.  Do this before the heavy-atom check in both the
    # RDKit and fallback paths; otherwise NaCl is incorrectly reported merely
    # as a molecule that is too small.
    inorganic_ions = ("[Na+]", "[Cl-]", "[K+]", "[Fe", "[Ca+2]", "[Mg+2]")
    if any(ion in clean_smi for ion in inorganic_ions):
        return {
            "status": "rejected",
            "passed": False,
            "code": "INORGANIC_SALT_NOT_DRUGLIKE",
            "reason": (
                f"SMILES '{clean_smi}' is an inorganic salt or isolated "
                "counter-ion, outside the small-molecule therapeutic domain."
            ),
            "recommendation": "abstain",
        }

    # 1. Chemical Valency & Parsing Check
    if RDKIT_AVAILABLE and Chem is not None:
        if mol is None:
            mol = Chem.MolFromSmiles(clean_smi)
        if mol is None:
            return {
                "status": "rejected",
                "passed": False,
                "code": "INVALID_CHEMICAL_STRUCTURE",
                "reason": f"SMILES '{clean_smi}' violates chemical valence or formatting rules.",
                "recommendation": "abstain",
            }
        assert Descriptors is not None and Crippen is not None
        num_heavy = mol.GetNumHeavyAtoms()
        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
    else:
        # Pure-Python valence / formatting sanity check
        num_heavy = max(len(re.findall(r"[A-Za-z]", clean_smi)), 1)
        props = _estimate_physicochemical_properties_from_smiles(clean_smi)
        mw = props["mw"]
        logp = props["logp"]

    # 2. Physics & Biological Boundary Check
    if num_heavy < 3:
        return {
            "status": "rejected",
            "passed": False,
            "code": "OUT_OF_BIOLOGICAL_DOMAIN",
            "reason": f"Molecule has only {num_heavy} heavy atoms; too small to serve as an authentic drug.",
            "recommendation": "abstain",
        }

    if mw < 50.0 or mw > 1500.0:
        return {
            "status": "warning_ood",
            "passed": False,
            "code": "OUT_OF_PHYSICAL_WEIGHT_RANGE",
            "reason": f"Molecular weight ({mw:.1f} g/mol) is outside standard therapeutic drug boundaries [50, 1500].",
            "recommendation": "abstain_or_flag_high_uncertainty",
        }

    if logp < -7.0 or logp > 10.0:
        return {
            "status": "warning_ood",
            "passed": False,
            "code": "EXTREME_LIPOPHILICITY",
            "reason": f"Calculated LogP ({logp:.2f}) represents extreme physical insolubility or non-permeation.",
            "recommendation": "abstain_or_flag_high_uncertainty",
        }

    return {
        "status": "grounded_valid",
        "passed": True,
        "code": "PHYSICOCHEMICALLY_SOUND",
        "reason": "Chemical structure and physical properties satisfy drug-like domain requirements.",
        "recommendation": "proceed",
    }


def compute_single_drug_admet(
    smiles: str,
    drug_name: str | None = None,
) -> dict[str, Any]:
    """Compute end-to-end ADMET, physicochemical parameters, and human danger indicators."""
    clean_smi = smiles.strip()
    guardrail = check_physicochemical_guardrails(clean_smi)

    if not guardrail["passed"] and guardrail["code"] in ("INVALID_CHEMICAL_STRUCTURE", "INORGANIC_SALT_NOT_DRUGLIKE"):
        return {
            "smiles": clean_smi,
            "drug_name": drug_name or "Unknown",
            "guardrail": guardrail,
            "status": "abstain_invalid_input",
            "danger_level": "UNKNOWN_UNEVALUATABLE",
            "admet": {},
            "toxicology": {},
        }

    mol = None
    if RDKIT_AVAILABLE and Chem is not None:
        mol = Chem.MolFromSmiles(clean_smi)

    # 1. Physicochemical Properties
    if mol is not None and Descriptors is not None and Crippen is not None and Lipinski is not None and rdMolDescriptors is not None:
        mw = float(Descriptors.MolWt(mol))
        logp = float(Crippen.MolLogP(mol))
        tpsa = float(Descriptors.TPSA(mol))
        hbd = int(Lipinski.NumHDonors(mol))
        hba = int(Lipinski.NumHAcceptors(mol))
        rotb = int(Lipinski.NumRotatableBonds(mol))
        aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
        heavy_atoms = mol.GetNumHeavyAtoms()
        fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    else:
        props = _estimate_physicochemical_properties_from_smiles(clean_smi)
        mw = props["mw"]
        logp = props["logp"]
        tpsa = props["tpsa"]
        hbd = props["hbd"]
        hba = props["hba"]
        rotb = props["rotb"]
        aromatic_rings = props["aromatic_rings"]
        heavy_atoms = props["heavy_atoms"]
        fsp3 = props["fsp3"]

    # 2. Rule of 5 (Lipinski) & Veber Compliance
    lipinski_violations = sum([
        mw > 500.0,
        logp > 5.0,
        hbd > 5,
        hba > 10,
    ])
    veber_compliant = (rotb <= 10) and (tpsa <= 140.0)

    # 3. Absorption (A)
    hia_high = (tpsa <= 130.0) and (-0.5 <= logp <= 5.5) and (mw <= 600.0)
    caco2_permeability = "High" if (tpsa < 80.0 and logp > 1.0) else "Moderate" if (tpsa <= 140.0) else "Low"

    # 4. Distribution (D)
    log_fu = -0.5 - 0.45 * max(logp, 0.0)
    f_unbound = max(min(math.exp(log_fu), 0.99), 0.01)

    # Blood-Brain Barrier (BBB) permeability score (Clark method approximation)
    bbb_score = 0.152 * logp - 0.0148 * tpsa + 0.139
    bbb_permeable = bbb_score > 0.0

    # 5. Metabolism (M) - CYP450 Family Affinities
    cyp_scores: dict[str, float] = {}
    if mol is not None:
        try:
            from src.data_prep.biophysical_engine import compute_cyp_affinities
        except ModuleNotFoundError:
            from data_prep.biophysical_engine import compute_cyp_affinities
        cyp_scores = compute_cyp_affinities(mol)
    else:
        cyp_scores = _estimate_cyp_profile_from_smiles(clean_smi, mw, logp)

    top_cyp = max(cyp_scores.items(), key=lambda x: x[1])

    # 6. Excretion (E)
    primary_clearance = "Renal Excretion" if (mw < 350.0 and logp < 1.0) else "Hepatic CYP Metabolism"

    # 7. Toxicophore Alerts & Toxicological Safety
    alerts_found: list[dict[str, str]] = []
    if mol is not None and Chem is not None:
        for category, rules in TOXICOPHORE_ALERTS.items():
            for name, smarts_str, desc in rules:
                pat = Chem.MolFromSmarts(smarts_str)
                if pat and mol.HasSubstructMatch(pat):
                    alerts_found.append({
                        "category": category,
                        "alert_name": name,
                        "description": desc,
                    })
    else:
        # Pure Python toxicophore substring heuristics
        s = clean_smi.lower()
        if "o=c1c=cc(=o)c=c1" in s or "c1ccc(o)cc1" in s:
            alerts_found.append({"category": "hepatotoxicity_reactive", "alert_name": "phenol_or_quinone_precursor", "description": "Potential reactive quinone/metabolite liability"})
        if "ccn(cc)" in s and "c1ccccc1" in s:
            alerts_found.append({"category": "cardiotoxicity_herg", "alert_name": "lipophilic_basic_amine", "description": "Potential hERG channel interaction motif"})

    # Check for Narrow Therapeutic Index (NTI)
    is_nti = False
    name_clean = drug_name.strip().lower() if drug_name else ""
    if name_clean in NARROW_THERAPEUTIC_INDEX_DRUGS:
        is_nti = True
    # Coumarin anticoagulant structure detection (Warfarin)
    if "c1c(o)c2ccccc2oc1=o" in clean_smi.lower() or "c2ccccc2oc1=o" in clean_smi.lower():
        is_nti = True

    # 8. Human Danger Level & Clinical Risk Factor
    herg_alerts = [a for a in alerts_found if a["category"] == "cardiotoxicity_herg"]
    hepato_alerts = [a for a in alerts_found if a["category"] == "hepatotoxicity_reactive"]

    danger_score = 0
    danger_factors = []

    if is_nti:
        danger_score += 4
        danger_factors.append("Narrow Therapeutic Index medication (small concentration shifts risk toxicity)")
    if herg_alerts:
        danger_score += 2
        danger_factors.append("hERG channel cardiotoxicity structural alert (risk of QT prolongation / arrhythmia)")
    if hepato_alerts:
        danger_score += 2
        danger_factors.append("Reactive metabolite structural alert (potential covalent liver binding / hepatotoxicity)")
    if lipinski_violations >= 2:
        danger_score += 1
        danger_factors.append("Poor drug-likeness (>1 Lipinski Rule of 5 violations)")
    if f_unbound < 0.05:
        danger_score += 1
        danger_factors.append("Extremely high plasma protein binding (<5% free drug; susceptible to displacement)")

    if danger_score >= 4:
        danger_level = "HIGH_CAUTION"
    elif danger_score >= 2:
        danger_level = "MODERATE_CAUTION"
    else:
        danger_level = "LOW_INTRINSIC_HAZARD"

    return {
        "smiles": clean_smi,
        "drug_name": drug_name or "Unknown Compound",
        "guardrail": guardrail,
        "physicochemical": {
            "molecular_weight": round(mw, 2),
            "logp": round(logp, 2),
            "tpsa": round(tpsa, 2),
            "h_bond_donors": hbd,
            "h_bond_acceptors": hba,
            "rotatable_bonds": rotb,
            "aromatic_rings": aromatic_rings,
            "heavy_atoms": heavy_atoms,
            "fraction_csp3": round(fsp3, 3),
            "lipinski_violations": lipinski_violations,
            "veber_compliant": veber_compliant,
            "oral_bioavailability_druglike": lipinski_violations <= 1 and veber_compliant,
        },
        "admet": {
            "absorption": {
                "human_intestinal_absorption": "High" if hia_high else "Moderate/Low",
                "human_intestinal_absorption_score": 0.90 if hia_high else 0.45,
                "caco2_permeability_estimate": caco2_permeability,
            },
            "distribution": {
                "fraction_unbound_plasma": round(f_unbound, 4),
                "plasma_protein_binding_pct": round((1.0 - f_unbound) * 100.0, 1),
                "blood_brain_barrier_permeable": bbb_permeable,
                "blood_brain_barrier_penetrant": bbb_permeable,
                "cns_access_category": "CNS Penetrant" if bbb_permeable else "Non-CNS Restricted",
            },
            "metabolism": {
                "primary_cyp_enzyme": top_cyp[0],
                "primary_cyp_affinity": round(top_cyp[1], 3),
                "cyp_affinity_profile": {k: round(v, 3) for k, v in cyp_scores.items()},
            },
            "excretion": {
                "primary_clearance_pathway": primary_clearance,
                "primary_clearance_route": primary_clearance,
            },
        },
        "toxicology": {
            "danger_level": danger_level,
            "danger_score": round(danger_score / 6.0, 2),
            "is_narrow_therapeutic_index": is_nti,
            "narrow_therapeutic_index": is_nti,
            "structural_alerts_count": len(alerts_found),
            "structural_alerts": alerts_found,
            "toxicophore_alerts": [f"{a['alert_name']}: {a['description']}" for a in alerts_found],
            "danger_factors": danger_factors,
        },
    }


class ADMETEngine:
    """First-principles ADMET, Toxicity & Guardrail Reasoning Engine."""

    def __init__(self) -> None:
        pass

    def analyze_drug(self, smiles: str, drug_name: str | None = None) -> dict[str, Any]:
        return compute_single_drug_admet(smiles, drug_name)

    def predict_cyp_affinities(self, smiles: str) -> dict[str, float]:
        res = compute_single_drug_admet(smiles)
        return res["admet"]["metabolism"]["cyp_affinity_profile"]

    def validate_guardrails(self, smiles: str) -> dict[str, Any]:
        return check_physicochemical_guardrails(smiles)
