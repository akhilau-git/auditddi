# Step 1.B: Task Formulation & Input–Output Contract — Technical Specification

---

## Executive Overview

Following the real-world clinical problem established in [Step 1.A](file:///d:/Drug-Drug%20Interaction/AuditDDI/docs/STEP_1A_REAL_WORLD_PROBLEM_DEFINITION.md), this document provides an exhaustive, field-by-field technical specification of **how the clinical problem is converted into an actionable Machine Learning contract** for **AuditDDI**:
1. **Mathematical Task Formulation**: Binary classification, symmetric permutation invariance, and multi-task loss.
2. **The 4 Generalization Regimes**: Transductive, S2 Cold-Start, S1 Cold-Start, and Scaffold-Disjoint.
3. **The Comprehensive Input Contract (Field-by-Field)**: Molecular graphs, fingerprints, enzymes, target sequences, 3D structures, and biophysical features.
4. **The Comprehensive Output Contract (Field-by-Field)**: Calibrated probabilities, decision thresholds, conformal sets, abstention flags, and substructure attribution.

---

## 1. Mathematical Task Formulation

### A. Primary Task: Permutation-Symmetric Binary Interaction Classification
Given any arbitrary pair of small-molecule drugs $(Drug_A, Drug_B)$, the system calculates the probability that co-prescribing them results in a clinically documented adverse Drug-Drug Interaction (DDI):

$$\hat{y}_{DDI} = P(Y = 1 \mid Drug_A, Drug_B) \in [0.0, 1.0]$$

Where the ground truth target $Y \in \{0, 1\}$ is defined as:
* **$Y = 1$ (Adverse Interaction Reported)**: The pair has documented real-world interaction evidence in surveillance databases (TWOSIDES) causing toxicity, adverse events, or severe pharmacokinetic clashes.
* **$Y = 0$ (Unreported Pair / Sampled Negative)**: The pair has no documented adverse interaction under equivalent surveillance thresholds. *(Note: As established in Step 1.A, $Y=0$ strictly means unobserved, not chemically proven safe).*

---

### B. The Permutation Invariance Principle (Commutative Law)
In clinical medicine, prescribing $Drug_A$ followed by $Drug_B$ is functionally identical to prescribing $Drug_B$ followed by $Drug_A$. 

If an AI model outputs $\hat{y}(A, B) = 0.85$ (High Risk) but $\hat{y}(B, A) = 0.20$ (Low Risk), it introduces a dangerous contradiction. In clinical reality, pills dissolve together in the bloodstream; the human body does not care which name was typed first into a computer.

To eliminate this flaw, **AuditDDI introduces an architectural guarantee of Permutation Symmetry (Order Invariance)**.

#### 1. The Flaw in Standard AI Models (Order Bias)
Traditional neural networks typically combine two items by placing them side-by-side in a long vector:
$$[\text{Drug A Features}, \text{Drug B Features}]$$
When the order is flipped to $[\text{Drug B Features}, \text{Drug A Features}]$, the neural network multiplies the features by completely different weight parameters. This creates an **artificial order bias**, causing the system to output conflicting risk scores for the exact same medical prescription.

#### 2. What AuditDDI Introduces: The 3 Commutative Operations
Instead of placing features side-by-side, AuditDDI combines the two drugs using **three commutative mathematical principles** (operations where the order of operands does not change the result):

* **A. Cumulative Systemic Burden (Symmetric Addition: $E_A + E_B$)**:
  * *The Principle*: Just as $5 + 3 = 3 + 5 = 8$, adding the two drug vectors together produces an identical result regardless of input order.
  * *Pharmacological Purpose*: It tells the model the **combined biological load** on the patient's body—such as the total amount of liver enzyme capacity required to process both medications.

* **B. Chemical & Target Contrast (Absolute Difference: $|E_A - E_B|$)**:
  * *The Principle*: Standard subtraction is order-dependent ($5 - 3 = +2$, but $3 - 5 = -2$). By taking the **absolute value** of the difference, the sign is always positive: $|5 - 3| = |3 - 5| = 2$.
  * *Pharmacological Purpose*: It captures the **divergence or contrast** between the two molecules—highlighting whether one drug is highly reactive or targets a biological pathway that the other does not.

* **C. Synergistic Clash / Collision (Multiplicative Cross-Product: $E_A \odot E_B$)**:
  * *The Principle*: Multiplication is inherently order-independent ($5 \times 3 = 3 \times 5 = 15$).
  * *Pharmacological Purpose*: It captures **direct biological competition**. If both drugs rely heavily on the exact same metabolic pathway (for example, both have high values for liver enzyme CYP3A4), multiplying those values together causes an exponential spike, instantly signaling a dangerous metabolic bottleneck.

#### 3. The Resulting Guarantee
By feeding only these three symmetric representations into the final prediction layers, AuditDDI ensures:
$$f(Drug_A, Drug_B) \equiv f(Drug_B, Drug_A) \quad \text{for every possible drug pair}$$
A doctor can query **(Warfarin, Aspirin)** or **(Aspirin, Warfarin)** and receive the **exact same risk probability, the exact same clinical verdict, and the exact same explanation**.

---

### C. Multi-Task Auxiliary Objective Function
Standard molecular Graph Neural Networks (GNNs) frequently suffer from **shortcut memorization**—they memorize high-frequency molecular hubs (e.g., common rings) rather than genuine pharmacology.

To eliminate this shortcut, AuditDDI incorporates an auxiliary multi-task regularizer using individual drug post-marketing adverse reaction data from the FDA FAERS database:

$$\mathcal{L}_{total} = \mathcal{L}_{DDI}(\hat{y}_{DDI}, y_{DDI}) + \lambda_{tox} \cdot \left[ \mathcal{L}_{BCE}(\hat{y}_{Tox_A}, y_{Tox_A}) + \mathcal{L}_{BCE}(\hat{y}_{Tox_B}, y_{Tox_B}) \right]$$

Where:
* $\mathcal{L}_{DDI}$ is the primary Binary Cross-Entropy with Logits loss on the drug pair interaction.
* $\mathcal{L}_{BCE}$ is the auxiliary loss on single-compound toxicity.
* $\lambda_{tox} = 0.3$ is the regularization weight balancing interaction learning with intrinsic chemical toxicity.

---

### D. The 4 Generalization Regimes

A model must never be evaluated solely on random splits. AuditDDI benchmarks across four distinct clinical and chemical regimes:

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ 1. Transductive Benchmark            │ 2. S2 Cold-Start (Semi-Inductive)    │
│ Both Drug A and Drug B exist in the  │ Exactly ONE drug is completely new.  │
│ training set; only their mutual pair │ Simulates adding an investigational  │
│ interaction is held out.             │ drug to an existing patient regimen. │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ 3. S1 Cold-Start (Fully Inductive)   │ 4. Bemis-Murcko Scaffold-Disjoint    │
│ BOTH Drug A and Drug B are novel and │ All core ring scaffolds in the test  │
│ excluded from training. Tests true   │ set are chemically disjoint from the │
│ extrapolation to new molecules.      │ training set. Zero scaffold leakage. │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 2. The Comprehensive Input Contract (Field-by-Field)

For any queried drug pair, AuditDDI ingests multimodal data across chemical, biological, and physical channels:

```
                            INPUT FEATURIZATION PIPELINE
[ Drug A SMILES ] ──► RDKit Graph Featurizer ──► Nodes: 44-dim, Edges: 14-dim
                  ──► Morgan Fingerprinter   ──► 1024-bit ECFP6 Vector
                  ──► PharmGKB Pathway Map   ──► 50-dim Multi-Hot Enzymes
                  ──► BindingDB Target Map   ──► 50-dim Affinity Profile
                  ──► UniProt / ESM-2        ──► 64-dim Amino Acid Embedding
                  ──► PDB Structure Cache    ──► 50-dim Co-crystal Pockets
                  ──► GEO Transcriptomics    ──► 2-dim Up/Down Regulation
                  ──► Biophysical Engine     ──► 12-dim ADMET & 11-dim Collision Terms
```

### Detailed Field Breakdown

#### 1. Molecular Graph: Atom Features ($x_i \in \mathbb{R}^{44}$)
Extracted per atom using RDKit ([`prepare_twosides.py`](file:///d:/Drug-Drug%20Interaction/AuditDDI/src/data_prep/prepare_twosides.py#L53-L75)):
* **Atomic Number (1–118)**: One-hot encoded across common pharmaceutical elements (C, N, O, S, P, F, Cl, Br, I, B, etc.).
* **Formal Charge ($-2, -1, 0, +1, +2$)**: Captures ionization states.
* **Hybridization**: One-hot ($sp, sp^2, sp^3, sp^3d, sp^3d^2$).
* **Aromaticity**: Binary flag ($1$ if the atom is in an aromatic system, $0$ otherwise).
* **Degree / Coordination**: Number of directly bonded neighboring atoms ($0$ to $6$).
* **Hydrogen Count**: Explicit hydrogens attached ($0$ to $4$).
* **Ring Membership**: Binary flag ($1$ if the atom belongs to any ring).

#### 2. Molecular Graph: Edge Features ($e_{ij} \in \mathbb{R}^{14}$)
Extracted per chemical bond between connected atoms:
* **Bond Order**: One-hot encoding (Single, Double, Triple, Aromatic).
* **Conjugation**: Binary flag ($1$ if the bond participates in conjugated pi-systems).
* **Stereochemistry**: One-hot encoding (None, Any, E/Trans, Z/Cis).
* **Ring Bond**: Binary flag ($1$ if the bond forms part of a ring structure).

#### 3. Global Chemical Fingerprint: 1,024-bit Morgan ECFP6
* **Algorithm**: Extended Connectivity Fingerprint with radius 3 (ECFP6) and chirality flags enabled.
* **Format**: Fixed-length $\{0, 1\}^{1024}$ binary bit vector.
* **Purpose**: Captures macro-scale circular substructures and functional groups that graph message passing might dilute across long atom paths.

#### 4. Pharmacogenomics: PharmGKB Enzymes ($\{0, 1\}^{50}$)
* **Format**: 50-dimensional multi-hot sparse binary vector.
* **Content**: Represents whether the drug interacts with primary human drug-metabolizing enzymes (CYP3A4, CYP2D6, CYP2C9, CYP1A2, CYP2C19, UGT glucuronosyltransferases, ABCB1 P-glycoprotein transporters).

#### 5. Target Affinities: BindingDB Profiles ($[0, 1]^{50}$)
* **Format**: 50-dimensional continuous vector normalized to $[0, 1]$.
* **Content**: Quantitative binding affinities ($K_i, K_d, IC_{50}$) across the top 50 human therapeutic drug targets (e.g., adrenergic receptors, dopamine receptors, COX enzymes, serotonin transporters).

#### 6. Macromolecular Protein Sequences: UniProt / ESM-2 ($\mathbb{R}^{64}$)
* **Input**: Raw primary amino-acid FASTA string of the drug's primary therapeutic target (e.g., CYP3A4 UniProt `P08684`: `"MALIPDLAMETWLLLAVSLVLLYLYGTRTHGLFK..."`).
* **Encoder**: Meta's **ESM-2** Transformer (`facebook/esm2_t6_8M_UR50D`) generating 320-dimensional contextualized residue vectors, length-invariant mean-pooled and projected to a 64-dimensional dense representation.
* **Fallback**: Standalone 1D-CNN over learned amino-acid embeddings for systems without transformer dependencies.

#### 7. 3D Macromolecular Complexes: PDB Pocket Signatures ($\mathbb{R}^{50}$)
* **Format**: 50-dimensional dense structural descriptor.
* **Content**: Derived from 3D co-crystal protein-ligand structures in the Protein Data Bank (PDB), capturing the geometric shape and volume of the drug's binding pocket.

#### 8. Transcriptomics: GEO Perturbation Signatures ($\mathbb{R}^{2}$)
* **Format**: 2-dimensional continuous scalar vector.
* **Content**: Gene Expression Omnibus (GEO) systemic RNA-seq perturbation metrics measuring cell-wide up-regulation and down-regulation following drug administration.

#### 9. First-Principles Biophysical & PK Collision Terms ($\mathbb{R}^{11}$)
Computed by [`biophysical_engine.py`](file:///d:/Drug-Drug%20Interaction/AuditDDI/src/data_prep/biophysical_engine.py#L46-L100):
* **5-Enzyme CYP Collision**: Element-wise product of Drug A and Drug B affinities across CYP3A4, CYP2D6, CYP2C9, CYP1A2, and CYP2C19.
* **Total Metabolic Clash Sum**: Cumulative sum of CYP enzyme conflicts.
* **Plasma Protein Binding Displacement Risk**: Calculated from unbound fractions as $(1 - f_{uA}) \times (1 - f_{uB})$.
* **Hepatic Clearance Co-dependence**: Overlap in liver clearance routes.
* **Physicochemical Deltas**: Molecular weight ratio $\frac{\min(MW_A, MW_B)}{\max(MW_A, MW_B)}$, lipophilicity delta $|\Delta \log P|$, and Polar Surface Area ratio $\frac{\min(TPSA_A, TPSA_B)}{\max(TPSA_A, TPSA_B)}$.

---

## 3. The Comprehensive Output Contract (Field-by-Field)

When inference is executed on a drug pair $(Drug_A, Drug_B)$, the system emits a validated, multi-layered clinical audit object:

```json
{
  "pair_metadata": {
    "drug_a_identifier": "Warfarin",
    "drug_b_identifier": "Aspirin",
    "drug_a_smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "drug_b_smiles": "CC(=O)Oc1ccccc1C(=O)O"
  },
  "risk_assessment": {
    "raw_model_score": 0.8921,
    "calibrated_probability": 0.8415,
    "calibration_method": "platt_logistic_regression",
    "clinical_decision": "HIGH_RISK_INTERACTION",
    "decision_threshold_applied": 0.3800
  },
  "uncertainty_and_safeguards": {
    "conformal_prediction_set": ["interaction"],
    "conformal_coverage_level": 0.90,
    "conformal_p_values": {
      "interaction_p_value": 0.012,
      "no_interaction_p_value": 0.884
    },
    "applicability_domain": {
      "drug_a_nearest_train_tanimoto": 0.724,
      "drug_b_nearest_train_tanimoto": 0.812,
      "pair_minimum_tanimoto": 0.724,
      "structural_ood_flag": false
    },
    "epistemic_uncertainty": {
      "mc_dropout_passes": 20,
      "epistemic_std": 0.034,
      "credible_interval_95": [0.774, 0.908],
      "high_uncertainty_flag": false
    },
    "safe_abstain": false,
    "safe_abstention_reason": "none"
  },
  "first_principles_pk_pd_collision": {
    "cyp_clearance_bottleneck": "CYP2C9_COMPETITIVE_SATURATION",
    "plasma_protein_binding_displacement_risk": 0.912,
    "narrow_therapeutic_index_alert": true,
    "organ_toxicophores_detected": [
      "cardiotoxicity_herg_absent",
      "hepatotoxicity_reactive_absent",
      "platelet_aggregation_inhibition_present"
    ]
  },
  "substructure_attribution": {
    "drug_a_salient_atom_indices": [12, 13, 14, 15],
    "drug_a_salient_motif": "4-hydroxycoumarin core",
    "drug_b_salient_atom_indices": [2, 3, 4],
    "drug_b_salient_motif": "acetylsalicylic acid carboxylate",
    "biological_explanation": "Aspirin irreversibly inhibits platelet COX-1 while displacing Warfarin from albumin; combined with Warfarin inhibition of VKORC1, this triggers extreme gastrointestinal hemorrhage risk."
  }
}
```

---

### Detailed Field Definitions

#### A. Block 1: `risk_assessment`
1. **`raw_model_score` (Float $\in [0.0, 1.0]$)**:
   * The uncalibrated neural network output directly following the final Sigmoid layer: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
   * *Limitation*: Deep neural networks produce uncalibrated overconfident scores; this raw score is logged for technical audit but never shown to physicians.
2. **`calibrated_probability` (Float $\in [0.0, 1.0]$)**:
   * The post-hoc calibrated probability score computed via Platt Scaling or Temperature Scaling fitted strictly on independent validation holdouts.
   * *Clinical Meaning*: If the model outputs `0.8415`, it means **statistically 84 out of 100 historical drug pairs assigned this score experienced a documented clinical interaction**.
3. **`calibration_method` (String)**:
   * Identifies the calibration mapping applied (`"platt_logistic_regression"` or `"temperature_scaling"`).
4. **`clinical_decision` (Enum: `HIGH_RISK_INTERACTION` | `LOW_RISK_BENIGN` | `ABSTAIN`)**:
   * The actionable categorical verdict delivered to the clinician.
5. **`decision_threshold_applied` (Float, Default: `0.3800`)**:
   * The optimized cutoff threshold $\tau^*$ derived on validation data via cost-sensitive Youden's Index ($J_{\text{cost}} = 2.0 \times \text{TPR} - \text{FPR}$).
   * If $\text{calibrated\_probability} \ge \tau^*$, the pair is classified as `HIGH_RISK_INTERACTION`.

---

#### B. Block 2: `uncertainty_and_safeguards`
1. **`conformal_prediction_set` (Array of Strings)**:
   * Finite-sample split-conformal inference set guaranteeing marginal coverage at $1 - \alpha = 0.90$:
     * `["interaction"]`: Statistically confident high risk.
     * `["no_interaction"]`: Statistically confident low risk.
     * `["no_interaction", "interaction"]`: **Ambiguous** — the model cannot distinguish between risk and safety with 90% confidence; triggers mandatory pharmacist review.
     * `["empty_set"]`: Out of distribution.
2. **`conformal_p_values` (Object)**:
   * Individual empirical p-values for both hypotheses testing whether the pair conforms to non-interacting or interacting validation distributions.
3. **`applicability_domain` (Object)**:
   * Measures structural similarity to the training set using Morgan fingerprint Tanimoto distance:
     * `drug_a_nearest_train_tanimoto`: Highest similarity score of Drug A to any compound in the training set.
     * `pair_minimum_tanimoto`: $\min(Tanimoto_A, Tanimoto_B)$.
     * `structural_ood_flag`: Set to `true` if $\min(Tanimoto_A, Tanimoto_B) < 0.40$.
4. **`epistemic_uncertainty` (Object)**:
   * Monte Carlo Dropout variance across 20 stochastic inference passes. An `epistemic_std` $> 0.10$ flags high internal model uncertainty.
5. **`safe_abstain` (Boolean)**:
   * Set to `true` if either compound is structurally OOD ($Tanimoto < 0.40$), if the conformal set is empty, or if MC Dropout uncertainty is excessive.
   * When `true`, the system refuses to output a decision and replies: **`"ABSTAIN: Out of Trusted Domain"`**.

---

#### C. Block 3: `first_principles_pk_pd_collision`
1. **`cyp_clearance_bottleneck` (String)**:
   * Identifies competitive metabolic clearance collisions in human liver Cytochrome P450 enzymes (e.g., `"CYP3A4_COMPETITIVE_SATURATION"`, `"CYP2C9_INHIBITION"`).
2. **`plasma_protein_binding_displacement_risk` (Float $\in [0.0, 1.0]$)**:
   * Mutual protein displacement risk calculated as $(1 - f_{uA}) \times (1 - f_{uB})$. Values $> 0.85$ indicate that both drugs bind heavily to albumin, creating a surge in free active drug concentration.
3. **`narrow_therapeutic_index_alert` (Boolean)**:
   * Automatically flags whether either compound belongs to the FDA Narrow Therapeutic Index (NTI) registry (e.g., Warfarin, Digoxin, Lithium, Theophylline, Tacrolimus) where small pharmacokinetic shifts cause lethality.

---

#### D. Block 4: `substructure_attribution`
1. **`drug_a_salient_atom_indices` / `drug_b_salient_atom_indices` (Array of Integers)**:
   * Exact atom indices within the molecular graph identified by GNNExplainer and Single-Component Occlusion as having the highest sensitivity impact on the risk score.
2. **`drug_a_salient_motif` / `drug_b_salient_motif` (String)**:
   * Human-readable chemical functional group name resolved from the salient atom indices (e.g., `"Coumarin core"`, `"Nitroaromatic group"`, `"Tertiary amine"`).
3. **`biological_explanation` (String)**:
   * A synthesized pharmacological mechanism connecting the chemical functional group, the liver enzyme or receptor, and the clinical outcome. This explains **why** the drugs interact, resolving the "black box" problem.
