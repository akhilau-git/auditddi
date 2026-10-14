# AuditDDI Research Model Card

## Model Overview
- **Model Name:** AuditDDI Multi-Label & Multimodal Interaction Framework
- **Architecture:** Modular Multi-Tower Framework (Edge-Aware GATv2, ECFP6 Projection Tower, ESM-2 Protein Language Models, openFDA CYP450 Profiles, and PharmGKB Gene Profiles) combined via Permutation-Invariant Symmetric Combinator $[E_A + E_B \parallel \lvert E_A - E_B\rvert \parallel E_A \odot E_B]$.
- **Task:** Multi-label adverse drug-drug interaction event estimation across ~863 standardized TWOSIDES event heads.
- **Intended Use:** Non-clinical research investigation of potential interaction signals, cold-start generalizability, and mechanistic audit trails.

---

## Benchmark Evidence ($n=5$ Matched Seeds: 0, 1, 2, 3, 4)

Evaluated across 5 independent random splits using paired $t$-tests and Benjamini-Hochberg (BH) False Discovery Rate control:

| Split Family | Test Partition | Baseline (ECFP) | Multimodal (ECFP+Target+Gene) | Lift | $p$-value | $q_{\text{BH}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bemis–Murcko Scaffold** | `test_s1` (Both Unseen) | 0.5214 | **0.5385** | **+0.0171** | **0.0007** | **0.0056** |
| **Bemis–Murcko Scaffold** | `test_s2` (One Unseen) | 0.6983 | **0.7162** | **+0.0179** | **0.0005** | **0.0056** |
| **Cold Drug** | `test_s1` (Both Unseen) | 0.5356 | **0.5510** | **+0.0154** | **0.0097** | **0.0350** |
| **Cold Drug** | `test_s2` (One Unseen) | 0.6984 | **0.7128** | **+0.0143** | **0.0022** | **0.0114** |

### Biological Stratum Ablation
- **Target Annotated (`bio2`)**: Scaffold S1 paired lift is **$+0.0222$ AUROC** ($p = 0.0009$, $q = 0.0056$).
- **Unannotated (`bio0`)**: Scaffold S1 paired difference is **$+0.0001$ AUROC** ($p = 0.98$).
- **Conclusion**: Performance gains on novel chemotypes are driven strictly by biological target mechanisms rather than random regularization artifacts.

---

## Known Boundaries & Scientific Limitations

1. **Cold-Start Prediction on Novel Chemotypes is Difficult**:
   While target features provide a statistically verified gain ($p < 0.001$), overall S1 AUROC remains modest (~0.54–0.55). Models must not be claimed to reliably "solve" zero-shot interaction prediction.
2. **Observational Nature of Supervision**:
   TWOSIDES represents observational pharmacovigilance reports from FAERS, not clinical ground truth. Sampled unobserved pairs mean "unreported", not clinically confirmed safe.
3. **No Patient Context**:
   The model does not accept or process patient dosage, genetics, comorbidities, organ clearance, or treatment timing.
4. **Regime-Aware Conformal Safety**:
   Standard pooled conformal prediction fails under distribution shift (coverage drops to 29%–73%). Regime-aware calibration repairs empirical coverage to **83%–92%** with explicit set-size and abstention flags.

---

## Research-Only Statement

> **AuditDDI is a research prototype.** Outputs are model estimates and source evidence for research review, not medical diagnoses, prescribing advice, or confirmed safety guarantees.
