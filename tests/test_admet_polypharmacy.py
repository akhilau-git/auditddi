"""Comprehensive Unit and Integration Tests for ADMET, Guardrails, and Polypharmacy Engine.

Validates:
1. Single-drug physicochemical, ADMET, and human danger indicators.
2. Multi-stage chemistry/physics guardrail checks and OOD abstention.
3. Pairwise structural-signal evaluation without unsupported clinical decisions.
4. Multi-drug polypharmacy cumulative metabolic bottleneck analysis.
"""

from __future__ import annotations

import unittest
from src.data_prep.admet_engine import (
    ADMETEngine,
    compute_single_drug_admet,
    check_physicochemical_guardrails,
)
from src.data_prep.polypharmacy_engine import (
    analyze_polypharmacy_regimen,
    format_polypharmacy_markdown,
)
from src.evaluation.clinical_audit_report import (
    generate_clinical_audit_report,
    generate_polypharmacy_audit_report,
)


class TestADMETAndPolypharmacyEngine(unittest.TestCase):
    """Test suite for biophysical ADMET, safety guardrails, and polypharmacy reasoning."""

    def setUp(self) -> None:
        self.engine = ADMETEngine()
        self.aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        self.paracetamol = "CC(=O)Nc1ccc(O)cc1"
        self.amoxicillin = "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"
        self.warfarin = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"
        self.fluconazole = "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F"
        self.procaine = "CCN(CC)CCOC(=O)c1ccc(N)cc1"

    def test_single_drug_admet_aspirin(self) -> None:
        prof = self.engine.analyze_drug(self.aspirin, "Aspirin")
        self.assertEqual(prof["guardrail"]["status"], "grounded_valid")
        self.assertAlmostEqual(prof["physicochemical"]["molecular_weight"], 180.16, delta=20.0)
        self.assertIn("CYP2C9", prof["admet"]["metabolism"]["cyp_affinity_profile"])
        self.assertIn(prof["admet"]["absorption"]["human_intestinal_absorption"], ["High", "Moderate/Low"])
        self.assertFalse(prof["toxicology"]["is_narrow_therapeutic_index"])

    def test_single_drug_nti_detection(self) -> None:
        # Warfarin should be recognized as a Narrow Therapeutic Index drug by structure and name
        prof_named = self.engine.analyze_drug(self.warfarin, "Warfarin")
        self.assertTrue(prof_named["toxicology"]["is_narrow_therapeutic_index"])
        self.assertIn("HIGH_CAUTION", prof_named["toxicology"]["danger_level"])

        # By structure alone (without explicit name)
        prof_unnamed = self.engine.analyze_drug(self.warfarin)
        self.assertTrue(prof_unnamed["toxicology"]["is_narrow_therapeutic_index"])

    def test_physicochemical_guardrails(self) -> None:
        # 1. Valid molecule
        valid_res = check_physicochemical_guardrails(self.paracetamol)
        self.assertTrue(valid_res["passed"])
        self.assertEqual(valid_res["code"], "PHYSICOCHEMICALLY_SOUND")

        # 2. Empty SMILES
        empty_res = check_physicochemical_guardrails("")
        self.assertFalse(empty_res["passed"])
        self.assertEqual(empty_res["code"], "EMPTY_INPUT")

        # 3. Inorganic salt (NaCl)
        salt_res = check_physicochemical_guardrails("[Na+].[Cl-]")
        self.assertFalse(salt_res["passed"])
        self.assertIn(salt_res["code"], ["INORGANIC_SALT_NOT_DRUGLIKE", "INVALID_CHEMICAL_STRUCTURE"])

        # 4. Tiny single atom
        tiny_res = check_physicochemical_guardrails("C")
        self.assertFalse(tiny_res["passed"])
        self.assertIn(tiny_res["code"], ["OUT_OF_BIOLOGICAL_DOMAIN", "OUT_OF_PHYSICAL_WEIGHT_RANGE"])

    def test_clinical_audit_report_safe_pair(self) -> None:
        # Paracetamol + Amoxicillin have orthogonal clearance
        report = generate_clinical_audit_report(self.paracetamol, self.amoxicillin)
        self.assertIn("[NO STRONG STRUCTURAL", report["decision"]["status"])
        self.assertIn("not evidence", report["decision"]["recommendation"])
        self.assertLess(report["metabolic_collision"]["collision_index"], 0.45)
        self.assertIn("# [AuditDDI Clinical & Biophysical Safety Dossier]", report["markdown"])

    def test_clinical_audit_report_high_risk_collision(self) -> None:
        # Warfarin + Fluconazole exhibit intense mutual CYP2C9 inhibition & NTI risk
        report = generate_clinical_audit_report(self.warfarin, self.fluconazole)
        self.assertIn("[HIGH PRIORITY", report["decision"]["status"])
        self.assertEqual(report["metabolic_collision"]["dominant_cyp"], "CYP2C9")
        self.assertGreater(report["metabolic_collision"]["collision_index"], 0.50)
        self.assertIn("CYP2C9", report["decision"]["recommendation"])

    def test_clinical_audit_report_guardrail_abstention(self) -> None:
        # Inorganic salt must cause model abstention rather than guessing
        report = generate_clinical_audit_report("[Na+].[Cl-]", self.aspirin)
        self.assertEqual(report["status"], "ABSTAIN_OUT_OF_DISTRIBUTION")
        self.assertIn("ABSTAIN_OUT_OF_DISTRIBUTION", report["decision"]["status"])
        self.assertIn("violates pure chemical/physical laws", report["decision"]["recommendation"])

    def test_polypharmacy_regimen_analysis(self) -> None:
        # 3-drug regimen: Warfarin + Fluconazole + Procaine
        drugs = [
            {"smiles": self.warfarin, "name": "Warfarin"},
            {"smiles": self.fluconazole, "name": "Fluconazole"},
            {"smiles": self.procaine, "name": "Procaine"},
        ]
        regimen = analyze_polypharmacy_regimen(drugs)
        self.assertEqual(regimen["num_drugs"], 3)
        self.assertEqual(regimen["overall_verdict"], "HIGH_PRIORITY_RESEARCH_SIGNAL")
        self.assertEqual(regimen["danger_level"], "RESEARCH_PRIORITY_HIGH")
        self.assertIn("CYP2C9", regimen["regimen_metrics"]["top_metabolic_bottleneck"])
        self.assertGreaterEqual(len(regimen["pairwise_interactions"]), 3)

        # Render markdown
        md = format_polypharmacy_markdown(regimen)
        self.assertIn("# [AuditDDI Multi-Drug Polypharmacy Safety Audit]", md)
        self.assertIn("Warfarin + Fluconazole", md)

    def test_polypharmacy_insufficient_drugs(self) -> None:
        # Regimen with only 1 drug must be rejected
        res = analyze_polypharmacy_regimen([self.aspirin])
        self.assertEqual(res["status"], "error_insufficient_drugs")

    def test_generate_polypharmacy_audit_report_wrapper(self) -> None:
        smiles_list = [self.aspirin, self.paracetamol, self.amoxicillin]
        names = ["Aspirin", "Paracetamol", "Amoxicillin"]
        out = generate_polypharmacy_audit_report(smiles_list, names)
        self.assertIn("regimen", out)
        self.assertIn("markdown", out)
        self.assertEqual(out["regimen"]["num_drugs"], 3)


if __name__ == "__main__":
    unittest.main()
