/**
 * AuditDDI: First-Principles Clinical Safety & Polypharmacy Web Application
 */

// API Configuration
const apiBaseUrl = new URL(
    document.body.dataset.apiBaseUrl || '/api',
    window.location.origin,
).toString().replace(/\/$/, '');

// Active State
let currentMode = 'audit'; // 'audit', 'poly', 'predict'
let currentWorkspace = 'research';
let authenticatedSession = null;
const latestReports = { audit: null, poly: null, predict: null };

const workspaceDefinitions = {
    research: {
        name: 'Research & Discovery', audience: 'For clinical researchers and pharmacologists',
        subtitle: 'Structure-led interaction research, mechanistic evidence and experimental follow-up.',
        modes: ['audit', 'poly', 'predict'],
        boundary: 'Research output only. Use it to prioritize experimental validation; it is not a prescribing decision.'
    },
    prescriber_review: {
        name: 'Prescriber Review', audience: 'For licensed prescribers reviewing molecular evidence',
        subtitle: 'A concise evidence review for discussion alongside the patient record and clinical judgement.',
        modes: ['audit', 'predict'],
        boundary: 'No patient-specific recommendation is available: this model has no linked dose, patient, exposure or outcome data.'
    },
    medication_safety: {
        name: 'Medication Safety Review', audience: 'For pharmacists and medication-safety teams',
        subtitle: 'Regimen-level molecular evidence to support escalation and medication-safety review.',
        modes: ['audit', 'poly'],
        boundary: 'Do not use this output alone to dispense, substitute, or change a dose. Escalate material concerns to the prescribing team.'
    },
    governance: {
        name: 'Evidence & Model Governance', audience: 'For regulatory, quality and model-governance reviewers',
        subtitle: 'Traceable model output, uncertainty, applicability diagnostics and reproducible audit evidence.',
        modes: ['audit', 'predict'],
        boundary: 'This workspace records model evidence and abstentions. It does not establish regulatory approval or clinical validity.'
    },
    portfolio_screening: {
        name: 'Portfolio Screening', audience: 'For pharmaceutical R&D teams',
        subtitle: 'Early portfolio prioritization using molecular interaction and ADMET signals.',
        modes: ['audit', 'poly', 'predict'],
        boundary: 'Use ranked outputs to plan assays, not as proof of safety, efficacy, or commercial suitability.'
    },
    public_information: {
        name: 'Medication Information', audience: 'For people seeking understandable medication-safety information',
        subtitle: 'Clear limits and the right next step when personalised medication review is needed.',
        modes: [],
        boundary: 'AuditDDI cannot check personal medication safety. A pharmacist or prescriber should review your medicines, doses, conditions and symptoms.'
    },
};

// DOM Elements - Navigation & Forms
const tabAuditBtn = document.getElementById('tab-btn-audit');
const tabPolyBtn = document.getElementById('tab-btn-poly');
const tabPredictBtn = document.getElementById('tab-btn-predict');

const formAudit = document.getElementById('form-mode-audit');
const formPoly = document.getElementById('form-mode-poly');
const formPredict = document.getElementById('form-mode-predict');

const emptyState = document.getElementById('empty-state');
const resultsAudit = document.getElementById('results-audit');
const resultsPoly = document.getElementById('results-poly');
const resultsPredict = document.getElementById('results-predict');
const errorState = document.getElementById('error-state');

// Mode Switcher
function switchMode(mode) {
    currentMode = mode;

    // Update Tab Buttons
    tabAuditBtn.classList.toggle('active', mode === 'audit');
    tabPolyBtn.classList.toggle('active', mode === 'poly');
    tabPredictBtn.classList.toggle('active', mode === 'predict');

    // Toggle Form Panels
    formAudit.classList.toggle('hidden', mode !== 'audit');
    formPoly.classList.toggle('hidden', mode !== 'poly');
    formPredict.classList.toggle('hidden', mode !== 'predict');

    // Hide Errors on mode switch
    errorState.classList.add('hidden');
}

function selectWorkspace(workspaceId) {
    if (authenticatedSession?.role && authenticatedSession.role !== 'admin') {
        workspaceId = authenticatedSession.role;
    }
    const workspace = workspaceDefinitions[workspaceId] || workspaceDefinitions.research;
    currentWorkspace = workspaceId in workspaceDefinitions ? workspaceId : 'research';
    document.getElementById('workspace-select').value = currentWorkspace;
    document.getElementById('workspace-name').innerText = workspace.name;
    document.getElementById('workspace-audience').innerText = workspace.audience;
    document.getElementById('workspace-subtitle').innerText = workspace.subtitle;
    document.getElementById('workspace-boundary').innerText = workspace.boundary;

    const tabs = { audit: tabAuditBtn, poly: tabPolyBtn, predict: tabPredictBtn };
    Object.entries(tabs).forEach(([mode, tab]) => {
        tab.classList.toggle('hidden', !workspace.modes.includes(mode));
    });
    const isPublicInformation = currentWorkspace === 'public_information';
    document.querySelector('.preset-bar').classList.toggle('hidden', isPublicInformation);
    document.querySelector('.app-container').classList.toggle('workspace-unavailable', isPublicInformation);
    document.getElementById('public-information-panel').classList.toggle('hidden', !isPublicInformation);
    document.querySelector('.input-section').classList.toggle('hidden', isPublicInformation);
    document.querySelector('.results-section').classList.toggle('hidden', isPublicInformation);

    if (!workspace.modes.includes(currentMode)) {
        if (workspace.modes.length) switchMode(workspace.modes[0]);
        else resetForm();
    }
    window.history.replaceState(null, '', `#workspace=${encodeURIComponent(currentWorkspace)}`);
}

function renderAuthentication(session) {
    const panel = document.getElementById('auth-panel');
    const loginForm = document.getElementById('login-form');
    const signedInView = document.getElementById('signed-in-view');
    const workspaceSelect = document.getElementById('workspace-select');
    if (!session.authentication_enabled) {
        panel.classList.add('hidden');
        authenticatedSession = null;
        workspaceSelect.disabled = false;
        return;
    }
    panel.classList.remove('hidden');
    authenticatedSession = session.user;
    loginForm.classList.toggle('hidden', Boolean(session.user));
    signedInView.classList.toggle('hidden', !session.user);
    workspaceSelect.disabled = Boolean(session.user && session.user.role !== 'admin');
    if (session.user) {
        document.getElementById('signed-in-user').innerText = `${session.user.email} · ${workspaceDefinitions[session.user.role]?.name || session.user.role}`;
        selectWorkspace(session.user.role === 'admin' ? currentWorkspace : session.user.role);
    }
}

async function refreshAuthentication() {
    try {
        const response = await fetch(`${apiBaseUrl}/auth/session`);
        if (!response.ok) return;
        renderAuthentication(await response.json());
    } catch (_) {
        // A deployment without the optional first-party access service remains usable.
    }
}

async function loginToWorkspace() {
    const error = document.getElementById('login-error');
    error.classList.add('hidden');
    const response = await fetch(`${apiBaseUrl}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            email: document.getElementById('login-email').value.trim(),
            password: document.getElementById('login-password').value,
        }),
    });
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        error.innerText = payload.detail || 'Unable to sign in.';
        error.classList.remove('hidden');
        return;
    }
    document.getElementById('login-password').value = '';
    await refreshAuthentication();
}

async function logoutFromWorkspace() {
    await fetch(`${apiBaseUrl}/auth/logout`, { method: 'POST' });
    await refreshAuthentication();
}

async function loadWorkspaceContract() {
    try {
        const response = await fetch(`${apiBaseUrl}/workspaces`);
        if (!response.ok) return;
        const payload = await response.json();
        (payload.workspaces || []).forEach((workspace) => {
            if (!workspaceDefinitions[workspace.id]) return;
            workspaceDefinitions[workspace.id].name = workspace.name;
            workspaceDefinitions[workspace.id].audience = `For ${workspace.audience.toLowerCase()}`;
            workspaceDefinitions[workspace.id].unavailableMessage = workspace.unavailable_message;
        });
    } catch (_) {
        // The shipped client-side copy preserves a useful, bounded interface
        // when the optional workspace metadata endpoint is unavailable.
    }
}

function downloadCurrentReport(kind) {
    const report = latestReports[kind];
    if (!report) {
        showError('Nothing to download yet', 'Run an analysis before downloading a structured report.');
        return;
    }
    const artifact = {
        generated_at: new Date().toISOString(),
        workspace: currentWorkspace,
        research_use_only: true,
        report,
    };
    const blob = new Blob([JSON.stringify(artifact, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `auditddi-${kind}-report.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
}

// Reset UI
function resetForm() {
    errorState.classList.add('hidden');
    resultsAudit.classList.add('hidden');
    resultsPoly.classList.add('hidden');
    resultsPredict.classList.add('hidden');
    emptyState.classList.remove('hidden');
}

// Global Loading State Handler
function setLoading(btnId, isLoading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');

    if (isLoading) {
        btn.disabled = true;
        if (btnText) btnText.classList.add('hidden');
        if (loader) loader.classList.remove('hidden');

        emptyState.classList.add('hidden');
        resultsAudit.classList.add('hidden');
        resultsPoly.classList.add('hidden');
        resultsPredict.classList.add('hidden');
        errorState.classList.add('hidden');
    } else {
        btn.disabled = false;
        if (btnText) btnText.classList.remove('hidden');
        if (loader) loader.classList.add('hidden');
    }
}

// Error State Display
function showError(title, message) {
    document.getElementById('error-title').innerText = title || 'Analysis Failed';
    document.getElementById('error-message').innerText = message || 'An error occurred during evaluation.';
    emptyState.classList.add('hidden');
    resultsAudit.classList.add('hidden');
    resultsPoly.classList.add('hidden');
    resultsPredict.classList.add('hidden');
    errorState.classList.remove('hidden');
}

function escapeHtml(value) {
    const element = document.createElement('div');
    element.textContent = String(value ?? '');
    return element.innerHTML;
}

// =========================================================================
// 1. CLINICAL PAIR AUDIT & ADMET (POST /api/audit/dossier)
// =========================================================================
async function runClinicalAudit() {
    const drugAName = document.getElementById('auditDrugAName').value.trim() || 'Drug A';
    const smilesA = document.getElementById('auditSmilesA').value.trim();
    const drugBName = document.getElementById('auditDrugBName').value.trim() || 'Drug B';
    const smilesB = document.getElementById('auditSmilesB').value.trim();

    if (!smilesA || !smilesB) {
        showError('Validation Error', 'Please provide valid SMILES strings for both drugs.');
        return;
    }

    setLoading('audit-submit-btn', true);

    try {
        const res = await fetch(`${apiBaseUrl}/audit/dossier`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                // Match the strict DDIRequest contract exposed by FastAPI.
                // Names remain client-side display labels; the research model
                // receives molecular structures only.
                smiles_a: smilesA,
                smiles_b: smilesB
            })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned error status HTTP ${res.status}`);
        }

        const data = await res.json();
        latestReports.audit = data;
        renderClinicalAuditDossier(data, drugAName, drugBName);
        setLoading('audit-submit-btn', false);
    } catch (err) {
        setLoading('audit-submit-btn', false);
        showError('Clinical Audit Failed', err.message);
    }
}

function renderClinicalAuditDossier(data, drugAName, drugBName) {
    const badge = document.getElementById('audit-status-badge');
    const decision = data.decision || {};
    const statusText = decision.status || data.status || 'UNKNOWN';

    // Set badge text & styling
    badge.innerText = statusText;
    badge.className = 'status-badge';

    if (statusText.includes('SAFE')) {
        badge.classList.add('badge-safe');
    } else if (statusText.includes('HIGH') || statusText.includes('CONTRAINDICATED')) {
        badge.classList.add('badge-danger');
    } else if (statusText.includes('MODERATE') || statusText.includes('MONITOR')) {
        badge.classList.add('badge-warning');
    } else if (statusText.includes('ABSTAIN')) {
        badge.classList.add('badge-abstain');
    }

    // Callout Details
    document.getElementById('audit-callout-title').innerText = statusText.includes('ABSTAIN') 
        ? 'First-Principles Physicochemical Refusal' 
        : 'Mechanistic Grounding & Proof';
    document.getElementById('audit-callout-rationale').innerText = decision.rationale || 'Detailed mechanistic profile computed.';
    document.getElementById('audit-callout-rec-text').innerText = decision.recommendation || 'Follow standard clinical protocols.';

    // Single-Drug ADMET Comparison Table
    const da = data.drug_a || {};
    const db = data.drug_b || {};
    document.getElementById('th-drug-a').innerText = da.name || drugAName;
    document.getElementById('th-drug-b').innerText = db.name || drugBName;

    const tbody = document.getElementById('audit-admet-body');
    tbody.innerHTML = '';

    const admetRows = [
        {
            param: 'Intrinsic Danger Level',
            valA: `<strong>${da.danger_level || 'UNKNOWN'}</strong> (${(da.danger_score || 0).toFixed(2)})`,
            valB: `<strong>${db.danger_level || 'UNKNOWN'}</strong> (${(db.danger_score || 0).toFixed(2)})`,
            desc: 'Narrow Therapeutic Index & intrinsic organ liability'
        },
        {
            param: 'Molecular Weight (MW)',
            valA: `${(da.mw || 0).toFixed(1)} g/mol`,
            valB: `${(db.mw || 0).toFixed(1)} g/mol`,
            desc: 'Small-molecule drug space (< 500 ideal)'
        },
        {
            param: 'Lipophilicity (LogP)',
            valA: `${(da.logp || 0).toFixed(2)}`,
            valB: `${(db.logp || 0).toFixed(2)}`,
            desc: 'Membrane partition & biological tissue accumulation'
        },
        {
            param: 'Polar Surface Area (TPSA)',
            valA: `${(da.tpsa || 0).toFixed(1)} Å²`,
            valB: `${(db.tpsa || 0).toFixed(1)} Å²`,
            desc: 'Membrane permeability (< 140 Å² ideal)'
        },
        {
            param: 'Human Intestinal Abs. (HIA)',
            valA: `${((da.hia_score || 0) * 100).toFixed(1)}%`,
            valB: `${((db.hia_score || 0) * 100).toFixed(1)}%`,
            desc: 'Predicted oral fraction absorbed'
        },
        {
            param: 'Unbound Fraction in Plasma (fu)',
            valA: `${((da.f_unbound || 0) * 100).toFixed(1)}%`,
            valB: `${((db.f_unbound || 0) * 100).toFixed(1)}%`,
            desc: 'Free active drug available for target binding'
        },
        {
            param: 'Blood-Brain Barrier (BBB)',
            valA: da.bbb_penetrant ? 'Penetrant (CNS Active)' : 'Non-penetrant',
            valB: db.bbb_penetrant ? 'Penetrant (CNS Active)' : 'Non-penetrant',
            desc: 'Central nervous system exposure liability'
        },
        {
            param: 'Primary Clearance Route',
            valA: `<strong>${da.clearance_route || 'Hepatic'}</strong>`,
            valB: `<strong>${db.clearance_route || 'Hepatic'}</strong>`,
            desc: 'Renal vs Hepatic metabolic excretion path'
        },
        {
            param: 'Dominant CYP Phase-I Enzyme',
            valA: `<strong>${da.top_cyp || 'CYP3A4'}</strong> (${(da.top_cyp_score || 0).toFixed(2)})`,
            valB: `<strong>${db.top_cyp || 'CYP3A4'}</strong> (${(db.top_cyp_score || 0).toFixed(2)})`,
            desc: 'Hepatic oxidation enzyme liability'
        },
        {
            param: 'Toxicophore Alerts',
            valA: (da.toxicophore_alerts && da.toxicophore_alerts.length) ? da.toxicophore_alerts.join(', ') : 'None detected',
            valB: (db.toxicophore_alerts && db.toxicophore_alerts.length) ? db.toxicophore_alerts.join(', ') : 'None detected',
            desc: 'Reactive quinone, epoxide, or mutagenic warheads'
        }
    ];

    admetRows.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${r.param}</strong></td>
            <td>${r.valA}</td>
            <td>${r.valB}</td>
            <td style="color: var(--text-muted); font-size: 0.78rem;">${r.desc}</td>
        `;
        tbody.appendChild(tr);
    });

    // Metabolic Collision Box
    const coll = data.metabolic_collision || {};
    const collIndex = coll.collision_index || 0;
    const cypOverlap = coll.cyp_overlap || 0;
    const domCYP = coll.dominant_cyp || 'CYP3A4';

    document.getElementById('audit-collision-index').innerText = collIndex.toFixed(3);
    document.getElementById('audit-dominant-cyp').innerText = domCYP;
    document.getElementById('audit-cyp-overlap').innerText = cypOverlap.toFixed(2);

    const bar = document.getElementById('audit-collision-bar');
    const barPct = Math.min(100, Math.max(0, collIndex * 100));
    bar.style.width = `${barPct}%`;
    bar.style.backgroundColor = collIndex >= 0.35 ? 'var(--danger)' : (collIndex >= 0.2 ? 'var(--warning)' : 'var(--success)');

    // Wet-Lab Validation Protocol
    document.getElementById('audit-wetlab-assay').innerText = decision.recommended_wet_lab_assay || 'Standard microsomal clearance assay';
    document.getElementById('audit-wetlab-equip').innerText = decision.analytical_equipment || 'LC-MS/MS System';

    // Show Audit Results View
    resultsAudit.classList.remove('hidden');
}

// =========================================================================
// 2. MULTI-DRUG POLYPHARMACY ANALYZER (POST /api/polypharmacy/analyze)
// =========================================================================
const DEFAULT_POLY_DRUGS = [
    { name: 'Simvastatin', smiles: 'CCC(C)(C)C(=O)OC1CC(C)C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O' },
    { name: 'Ketoconazole', smiles: 'CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OCC3COC(O3)(CN4C=CN=C4)C5=C(C=C(C=C5)Cl)Cl' },
    { name: 'Clarithromycin', smiles: 'CCC1C(C(C(N(C)C)CC(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)(C)O)N(C)C)C)O' }
];

function initPolyDrugList(initialDrugs = DEFAULT_POLY_DRUGS) {
    const list = document.getElementById('poly-drug-list');
    list.innerHTML = '';
    initialDrugs.forEach((drug, idx) => {
        addPolyDrugRow(drug.name, drug.smiles, idx + 1);
    });
}

function addPolyDrugRow(name = '', smiles = '', rowNum = null) {
    const list = document.getElementById('poly-drug-list');
    const index = rowNum || (list.children.length + 1);
    const row = document.createElement('div');
    row.className = 'poly-drug-row';
    row.innerHTML = `
        <div class="poly-drug-header">
            <span class="poly-drug-idx">Drug #${index}</span>
            ${index > 2 ? '<button type="button" class="remove-drug-btn" onclick="removePolyDrugRow(this)">✕ Remove</button>' : ''}
        </div>
        <div class="input-group" style="margin-bottom: 0.5rem;">
            <input type="text" class="poly-drug-name" placeholder="Drug Name (e.g. Drug ${index})" value="${name}" required autocomplete="off">
        </div>
        <div class="input-group">
            <input type="text" class="poly-drug-smiles" placeholder="SMILES string" value="${smiles}" required autocomplete="off">
        </div>
    `;
    list.appendChild(row);
}

function removePolyDrugRow(btn) {
    const row = btn.closest('.poly-drug-row');
    if (row) {
        row.remove();
        // Re-number rows
        const list = document.getElementById('poly-drug-list');
        Array.from(list.children).forEach((r, idx) => {
            const tag = r.querySelector('.poly-drug-idx');
            if (tag) tag.innerText = `Drug #${idx + 1}`;
        });
    }
}

async function runPolypharmacyAnalysis() {
    const list = document.getElementById('poly-drug-list');
    const rows = list.querySelectorAll('.poly-drug-row');
    const drugs = [];

    rows.forEach(r => {
        const name = r.querySelector('.poly-drug-name').value.trim() || 'Unknown Drug';
        const smiles = r.querySelector('.poly-drug-smiles').value.trim();
        if (smiles) {
            drugs.push({ name, smiles });
        }
    });

    if (drugs.length < 2) {
        showError('Validation Error', 'A polypharmacy regimen requires at least 2 drugs.');
        return;
    }

    setLoading('poly-submit-btn', true);

    try {
        const res = await fetch(`${apiBaseUrl}/polypharmacy/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ drugs })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned error status HTTP ${res.status}`);
        }

        const data = await res.json();
        latestReports.poly = data;
        renderPolypharmacyDossier(data);
        setLoading('poly-submit-btn', false);
    } catch (err) {
        setLoading('poly-submit-btn', false);
        showError('Polypharmacy Evaluation Failed', err.message);
    }
}

function renderPolypharmacyDossier(data) {
    // The API includes a Markdown rendering alongside the structured regimen.
    // Render the structured data so this UI remains independent of Markdown.
    const regimen = data.regimen || data;
    const badge = document.getElementById('poly-status-badge');
    const verdict = regimen.overall_verdict || 'EVALUATED';
    const danger = regimen.danger_level || 'UNKNOWN';

    badge.innerText = `${verdict} (${danger})`;
    badge.className = 'status-badge';

    if (verdict.includes('SAFE')) {
        badge.classList.add('badge-safe');
    } else if (verdict.includes('HIGH') || verdict.includes('CONTRAINDICATED')) {
        badge.classList.add('badge-danger');
    } else if (verdict.includes('MONITOR') || verdict.includes('CAUTION')) {
        badge.classList.add('badge-warning');
    } else if (verdict.includes('ABSTAIN')) {
        badge.classList.add('badge-abstain');
    }

    // Guidance callout
    document.getElementById('poly-guidance-text').innerText = regimen.clinical_guidance || 'Cumulative regimen evaluated.';

    // Cumulative CYP Saturation Gauges
    const metrics = regimen.regimen_metrics || {};
    const cypLoads = metrics.cumulative_cyp_loads || {};
    const cypGrid = document.getElementById('poly-cyp-grid');
    cypGrid.innerHTML = '';

    const targetCYPs = ['CYP3A4', 'CYP2D6', 'CYP2C9', 'CYP1A2', 'CYP2C19'];
    targetCYPs.forEach(cyp => {
        const load = cypLoads[cyp] || 0;
        const loadPct = Math.min(100, Math.max(0, (load / 2.0) * 100));

        let color = 'var(--success)';
        let status = 'NOMINAL';
        if (load >= 1.5) {
            color = 'var(--danger)';
            status = 'SATURATED';
        } else if (load >= 1.0) {
            color = 'var(--warning)';
            status = 'ELEVATED';
        }

        const card = document.createElement('div');
        card.className = 'cyp-gauge-card';
        card.innerHTML = `
            <div class="cyp-gauge-name">${cyp}</div>
            <div class="cyp-gauge-val" style="color: ${color};">${load.toFixed(2)}</div>
            <div class="cyp-bar-track">
                <div class="cyp-bar-fill" style="width: ${loadPct}%; background-color: ${color};"></div>
            </div>
            <div class="cyp-status-tag" style="color: ${color};">${status}</div>
        `;
        cypGrid.appendChild(card);
    });

    // Cumulative Organ Liabilities
    document.getElementById('poly-herg-score').innerText = metrics.cumulative_herg_cardiotoxicity_alerts || 0;
    document.getElementById('poly-hepato-score').innerText = metrics.cumulative_hepatotoxicity_alerts || 0;

    // Pairwise Interactions inside Regimen
    const pairsContainer = document.getElementById('poly-pairs-container');
    pairsContainer.innerHTML = '';
    const interactions = regimen.pairwise_interactions || [];

    interactions.forEach(pair => {
        const isSevere = pair.severe_interaction_risk;
        const reasons = Array.isArray(pair.mechanistic_reasons)
            ? pair.mechanistic_reasons.join('; ')
            : (pair.mechanistic_reasons || 'Orthogonal clearance paths');

        const pCard = document.createElement('div');
        pCard.className = 'poly-pair-card';
        pCard.innerHTML = `
            <div>
                <strong>${escapeHtml(pair.pair || `${pair.drug_a} + ${pair.drug_b}`)}</strong>
                <div style="color: var(--text-secondary); font-size: 0.78rem; margin-top: 0.2rem;">${escapeHtml(reasons)}</div>
            </div>
            <div>
                <span class="${isSevere ? 'pair-badge-severe' : 'pair-badge-ok'}">
                    ${isSevere ? 'SEVERE COLLISION' : 'COMPATIBLE'}
                </span>
            </div>
        `;
        pairsContainer.appendChild(pCard);
    });

    resultsPoly.classList.remove('hidden');
}

// =========================================================================
// 3. NEURAL NETWORK PREDICTOR (POST /api/predict)
// =========================================================================
async function checkRisk() {
    const smilesA = document.getElementById('smilesA').value.trim();
    const smilesB = document.getElementById('smilesB').value.trim();

    if (!smilesA || !smilesB) return;

    setLoading('predict-btn', true);

    try {
        const res = await fetch(`${apiBaseUrl}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles_a: smilesA, smiles_b: smilesB })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Prediction failed with HTTP ${res.status}`);
        }

        const data = await res.json();
        latestReports.predict = data;
        renderPredictResults(data, res);
        setLoading('predict-btn', false);
    } catch (e) {
        setLoading('predict-btn', false);
        showError('Prediction Failed', e.message || 'Could not connect to backend.');
    }
}

function renderPredictResults(data, res) {
    const threshold = typeof data.decision_threshold_used === 'number'
        ? data.decision_threshold_used
        : 0.5;
    const thresholdPct = threshold * 100;
    const riskPct = (typeof data.interaction_risk_estimate === 'number'
        ? data.interaction_risk_estimate
        : 0) * 100;
    const interactionPredicted = data.interaction_predicted;

    const riskValue = document.getElementById('risk-value');
    const riskBar = document.getElementById('risk-bar');
    const riskDesc = document.getElementById('risk-desc');

    riskValue.innerText = `${riskPct.toFixed(1)}%`;
    riskValue.className = `metric-value ${riskPct >= thresholdPct ? 'val-danger' : 'val-safe'}`;

    setTimeout(() => {
        riskBar.style.width = `${riskPct}%`;
        riskBar.style.backgroundColor = riskPct >= thresholdPct ? 'var(--danger)' : 'var(--success)';
    }, 100);

    if (interactionPredicted) {
        riskDesc.innerHTML = `<span style="color: var(--danger)">Interaction Predicted</span> (>= decision threshold ${thresholdPct.toFixed(1)}%).`;
    } else {
        riskDesc.innerHTML = `<span style="color: var(--success)">Sub-Threshold Score</span> (< threshold ${thresholdPct.toFixed(1)}%).`;
    }

    // Toxicity
    const toxAVal = (data.drug_a_toxicity && typeof data.drug_a_toxicity.score === 'number') ? data.drug_a_toxicity.score * 100 : 0;
    const toxBVal = (data.drug_b_toxicity && typeof data.drug_b_toxicity.score === 'number') ? data.drug_b_toxicity.score * 100 : 0;
    document.getElementById('toxA-value').innerText = `${toxAVal.toFixed(1)}%`;
    document.getElementById('toxB-value').innerText = `${toxBVal.toFixed(1)}%`;

    // Meta elements
    document.getElementById('meta-arch').innerText = data.model_architecture || 'Edge-Aware GNN';
    document.getElementById('meta-auroc').innerText = data.stored_validation_evidence?.auroc 
        ? `${data.stored_validation_evidence.auroc.toFixed(4)} (internal validation)` 
        : 'Unavailable';
    document.getElementById('meta-threshold').innerText = `${thresholdPct.toFixed(2)}% (${interactionPredicted ? 'Triggered' : 'Below Cutoff'})`;
    document.getElementById('meta-calib').innerText = data.score_calibration?.status || 'Uncalibrated';
    document.getElementById('meta-conformal').innerText = data.prediction_uncertainty?.prediction_set 
        ? `${data.prediction_uncertainty.prediction_set} (${data.prediction_uncertainty.abstain ? 'Abstain' : 'Confident'})` 
        : 'Not configured';
    document.getElementById('meta-ood').innerText = data.structural_applicability_domain?.outside_structural_domain 
        ? 'Out-of-Domain' 
        : 'In-Domain';
    document.getElementById('meta-reqid').innerText = res.headers.get('x-request-id') || 'Unavailable';

    resultsPredict.classList.remove('hidden');
}

// =========================================================================
// 4. QUICK SCENARIOS PRESET LOADER
// =========================================================================
function loadScenario(scenario) {
    if (scenario === 'safe') {
        switchMode('audit');
        document.getElementById('auditDrugAName').value = 'Amoxicillin';
        document.getElementById('auditSmilesA').value = 'CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C';
        document.getElementById('auditDrugBName').value = 'Acetaminophen';
        document.getElementById('auditSmilesB').value = 'CC(=O)NC1=CC=C(C=C1)O';
        runClinicalAudit();
    } else if (scenario === 'nti') {
        switchMode('audit');
        document.getElementById('auditDrugAName').value = 'Warfarin';
        document.getElementById('auditSmilesA').value = 'CC(=O)CC(C1=CC=CC=C1)C2=C(C(=O)OC3=CC=CC=C23)O';
        document.getElementById('auditDrugBName').value = 'Fluconazole';
        document.getElementById('auditSmilesB').value = 'C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O';
        runClinicalAudit();
    } else if (scenario === 'poly') {
        switchMode('poly');
        initPolyDrugList([
            { name: 'Simvastatin', smiles: 'CCC(C)(C)C(=O)OC1CC(C)C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O' },
            { name: 'Ketoconazole', smiles: 'CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OCC3COC(O3)(CN4C=CN=C4)C5=C(C=C(C=C5)Cl)Cl' },
            { name: 'Clarithromycin', smiles: 'CCC1C(C(C(N(C)C)CC(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)(C)O)N(C)C)C)O' }
        ]);
        runPolypharmacyAnalysis();
    } else if (scenario === 'ood') {
        switchMode('audit');
        document.getElementById('auditDrugAName').value = 'Lithium Chloride';
        document.getElementById('auditSmilesA').value = '[Li+].[Cl-]';
        document.getElementById('auditDrugBName').value = 'Aspirin';
        document.getElementById('auditSmilesB').value = 'CC(=O)OC1=CC=CC=C1C(=O)O';
        runClinicalAudit();
    }
}

// Initial setup on page load
document.addEventListener('DOMContentLoaded', () => {
    initPolyDrugList();
    const workspaceId = new URLSearchParams(window.location.hash.replace(/^#/, '')).get('workspace');
    selectWorkspace(workspaceId || 'research');
    loadWorkspaceContract().then(() => selectWorkspace(currentWorkspace));
    refreshAuthentication();
});
