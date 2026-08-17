/**
 * CardioGuard AI - Frontend Interactivity & API Integration
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initPresets();
    initAssessmentForm();
    initSimulator();
    initBatchUpload();
    initModelMetrics();

    // Auto-evaluate default form values on load
    const form = document.getElementById('patient-assessment-form');
    if (form) {
        submitAssessment(new FormData(form));
    }
});

/* ==================== 1. TAB NAVIGATION ==================== */
function initTabs() {
    const tabButtons = document.querySelectorAll('.nav-tab');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-tab');

            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }

            // Trigger chart resize if metrics tab selected
            if (targetId === 'metrics-tab' && window.featureChart) {
                window.featureChart.resize();
            }
        });
    });
}

/* ==================== 2. CLINICAL PRESETS ==================== */
function initPresets() {
    const lowRiskPreset = {
        age: 38,
        gender: 0,
        systolic_bp: 115,
        diastolic_bp: 74,
        cholesterol: 175,
        glucose: 85,
        bmi: 22.4,
        smoking: 0,
        alcohol_intake: 0,
        physical_activity: 1,
        resting_ecg: 0,
        max_heart_rate: 168,
        chest_pain_type: 3,
        exercise_angina: 0
    };

    const highRiskPreset = {
        age: 64,
        gender: 1,
        systolic_bp: 162,
        diastolic_bp: 98,
        cholesterol: 278,
        glucose: 165,
        bmi: 33.2,
        smoking: 1,
        alcohol_intake: 1,
        physical_activity: 0,
        resting_ecg: 2,
        max_heart_rate: 118,
        chest_pain_type: 0,
        exercise_angina: 1
    };

    document.getElementById('preset-low-risk')?.addEventListener('click', () => {
        fillForm(lowRiskPreset);
        submitAssessment(new FormData(document.getElementById('patient-assessment-form')));
    });

    document.getElementById('preset-high-risk')?.addEventListener('click', () => {
        fillForm(highRiskPreset);
        submitAssessment(new FormData(document.getElementById('patient-assessment-form')));
    });

    document.getElementById('btn-reset-form')?.addEventListener('click', () => {
        document.getElementById('patient-assessment-form')?.reset();
    });
}

function fillForm(data) {
    for (const [key, val] of Object.entries(data)) {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = val;
        }
    }
}

/* ==================== 3. ASSESSMENT FORM & GAUGE ==================== */
function initAssessmentForm() {
    const form = document.getElementById('patient-assessment-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        await submitAssessment(formData);
    });
}

async function submitAssessment(formData) {
    const payload = {};
    formData.forEach((value, key) => {
        payload[key] = key === 'bmi' ? parseFloat(value) : parseInt(value, 10);
    });

    const submitBtn = document.getElementById('btn-submit-assessment');
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Biomarkers...';
    }

    try {
        const response = await fetch('/api/v1/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Prediction failed');
        }

        const data = await response.json();
        renderAssessmentResults(data);
    } catch (err) {
        alert(`Assessment Error: ${err.message}`);
    } finally {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Evaluate Patient Risk Profile';
        }
    }
}

function renderAssessmentResults(data) {
    // 1. Classification & Tier Pill
    const classificationEl = document.getElementById('res-classification');
    const tierPill = document.getElementById('res-tier-pill');
    
    if (classificationEl) classificationEl.textContent = data.prediction_label;
    if (tierPill) {
        tierPill.textContent = data.risk_tier;
        tierPill.style.backgroundColor = `${data.risk_color}25`;
        tierPill.style.color = data.risk_color;
        tierPill.style.border = `1px solid ${data.risk_color}`;
    }

    // 2. SVG Gauge Arc
    // Arc circumference is ~251.2
    const percentage = data.risk_percentage;
    const maxOffset = 251.2;
    const currentOffset = maxOffset - (maxOffset * (percentage / 100));
    
    const gaugeArc = document.getElementById('gauge-arc');
    if (gaugeArc) {
        gaugeArc.style.strokeDashoffset = currentOffset;
        gaugeArc.style.stroke = data.risk_color;
    }

    // Numbers
    document.getElementById('res-percentage').textContent = `${percentage.toFixed(1)}%`;
    document.getElementById('res-confidence').textContent = `${data.confidence.toFixed(1)}%`;
    document.getElementById('res-probability').textContent = data.risk_probability.toFixed(4);

    // 3. XAI Factor Breakdown
    const xaiContainer = document.getElementById('res-xai-list');
    if (xaiContainer) {
        xaiContainer.innerHTML = '';
        data.risk_breakdown.forEach(item => {
            const badgeClass = item.impact_level === 'Critical' ? 'tag-critical' :
                               item.impact_level === 'High' ? 'tag-high' :
                               item.impact_level === 'Moderate' ? 'tag-moderate' : 'tag-low';

            const div = document.createElement('div');
            div.className = 'xai-item';
            div.innerHTML = `
                <div class="xai-item-top">
                    <div>
                        <span class="xai-feature-name">${item.display_name}: <strong>${item.patient_value}</strong></span>
                        <span style="font-size: 11px; color: var(--text-dim); margin-left: 6px;">(Ref: ${item.benchmark})</span>
                    </div>
                    <span class="xai-badge ${badgeClass}">${item.status}</span>
                </div>
                <div class="xai-item-desc">${item.description}</div>
                <div class="xai-progress-bar">
                    <div class="xai-progress-fill" style="width: ${item.relative_contribution_pct * 3.5}%; background-color: ${data.risk_color};"></div>
                </div>
            `;
            xaiContainer.appendChild(div);
        });
    }

    // 4. Clinical Recommendations
    const recsContainer = document.getElementById('res-recs-list');
    if (recsContainer) {
        recsContainer.innerHTML = '';
        data.recommendations.forEach(rec => {
            const div = document.createElement('div');
            div.className = `rec-item prio-${rec.priority}`;
            div.innerHTML = `
                <div class="rec-header">
                    <span class="rec-title"><i class="fa-solid fa-notes-medical"></i> ${rec.title}</span>
                    <span class="rec-prio">${rec.priority} Priority</span>
                </div>
                <div class="rec-action">${rec.action_item}</div>
                <div class="rec-rationale"><strong>Clinical Rationale:</strong> ${rec.clinical_rationale}</div>
            `;
            recsContainer.appendChild(div);
        });
    }
}

/* ==================== 4. WHAT-IF SIMULATOR ==================== */
function initSimulator() {
    const sysSlider = document.getElementById('sim-range-sys');
    const cholSlider = document.getElementById('sim-range-chol');
    const glucSlider = document.getElementById('sim-range-gluc');
    const bmiSlider = document.getElementById('sim-range-bmi');
    const smokeCheck = document.getElementById('sim-check-smoke');
    const activeCheck = document.getElementById('sim-check-active');

    if (!sysSlider) return;

    const updateSimulator = debounce(async () => {
        const sysVal = parseInt(sysSlider.value, 10);
        const cholVal = parseInt(cholSlider.value, 10);
        const glucVal = parseInt(glucSlider.value, 10);
        const bmiVal = parseFloat(bmiSlider.value);
        const isSmoker = smokeCheck.checked ? 1 : 0;
        const isActive = activeCheck.checked ? 1 : 0;

        document.getElementById('sim-val-sys').textContent = `${sysVal} mmHg`;
        document.getElementById('sim-val-chol').textContent = `${cholVal} mg/dL`;
        document.getElementById('sim-val-gluc').textContent = `${glucVal} mg/dL`;
        document.getElementById('sim-val-bmi').textContent = `${bmiVal}`;

        const payload = {
            age: 58,
            gender: 1,
            systolic_bp: sysVal,
            diastolic_bp: Math.round(sysVal * 0.65),
            cholesterol: cholVal,
            glucose: glucVal,
            bmi: bmiVal,
            smoking: isSmoker,
            alcohol_intake: 0,
            physical_activity: isActive,
            resting_ecg: 1,
            max_heart_rate: 135,
            chest_pain_type: 1,
            exercise_angina: 0
        };

        try {
            const res = await fetch('/api/v1/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) return;
            const data = await res.json();

            const scoreEl = document.getElementById('sim-res-score');
            const tierEl = document.getElementById('sim-res-tier');
            const summaryEl = document.getElementById('sim-impact-summary');

            scoreEl.textContent = `${data.risk_percentage.toFixed(1)}%`;
            scoreEl.style.color = data.risk_color;
            tierEl.textContent = data.risk_tier;
            tierEl.style.color = data.risk_color;

            let modText = [];
            if (sysVal <= 120) modText.push("Optimal Blood Pressure");
            if (cholVal <= 200) modText.push("Controlled Lipids");
            if (isSmoker === 0) modText.push("Non-Smoking Status");
            if (isActive === 1) modText.push("Regular Physical Activity");

            if (modText.length > 0) {
                summaryEl.innerHTML = `<strong>Favorable Risk Modifiers:</strong> ${modText.join(" • ")}. Simulated 10-year major adverse cardiac event probability is currently stratified at <strong>${data.risk_tier}</strong>.`;
            } else {
                summaryEl.innerHTML = `Multiple unmanaged cardiovascular risk factors detected. Consider lifestyle modifications to mitigate calculated hazard.`;
            }
        } catch (e) {
            console.error(e);
        }
    }, 150);

    [sysSlider, cholSlider, glucSlider, bmiSlider, smokeCheck, activeCheck].forEach(el => {
        el.addEventListener('input', updateSimulator);
    });

    updateSimulator();
}

/* ==================== 5. BATCH CSV UPLOAD ==================== */
function initBatchUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('batch-file-input');
    const browseBtn = document.getElementById('btn-browse-file');

    if (!dropZone || !fileInput) return;

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            processBatchFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            processBatchFile(fileInput.files[0]);
        }
    });
}

async function processBatchFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a valid .csv file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const dropZone = document.getElementById('drop-zone');
    dropZone.innerHTML = `
        <i class="fa-solid fa-spinner fa-spin upload-icon"></i>
        <h3 class="upload-title">Processing Patient Cohort...</h3>
        <p class="upload-subtitle">${file.name}</p>
    `;

    try {
        const response = await fetch('/api/v1/predict-batch', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Batch prediction failed');
        }

        const data = await response.json();
        renderBatchResults(data);
    } catch (err) {
        alert(`Batch Error: ${err.message}`);
    } finally {
        dropZone.innerHTML = `
            <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
            <h3 class="upload-title">Drag & Drop Patient CSV File</h3>
            <p class="upload-subtitle">or click to browse from your computer</p>
            <input type="file" id="batch-file-input" accept=".csv" class="file-input-hidden">
            <button class="btn btn-primary" id="btn-browse-file">
                <i class="fa-solid fa-folder-open"></i> Select CSV File
            </button>
        `;
        initBatchUpload();
    }
}

function renderBatchResults(data) {
    const container = document.getElementById('batch-results-container');
    container.classList.remove('hidden');

    document.getElementById('batch-total').textContent = data.summary.total_records.toLocaleString();
    document.getElementById('batch-high-risk').textContent = data.summary.high_risk_count.toLocaleString();
    document.getElementById('batch-low-risk').textContent = data.summary.low_risk_count.toLocaleString();
    document.getElementById('batch-pct').textContent = `${data.summary.high_risk_percentage.toFixed(1)}%`;

    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = '';

    data.preview.forEach((row, idx) => {
        const tr = document.createElement('tr');
        const tierClass = row.risk_tier === 'Critical Risk' ? 'tag-critical' :
                          row.risk_tier === 'High Risk' ? 'tag-high' :
                          row.risk_tier === 'Moderate Risk' ? 'tag-moderate' : 'tag-low';

        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>${row.age} yrs</td>
            <td>${row.gender === 1 ? 'Male' : 'Female'}</td>
            <td>${row.systolic_bp}/${row.diastolic_bp}</td>
            <td>${row.cholesterol} mg/dL</td>
            <td>${row.glucose} mg/dL</td>
            <td>${row.bmi}</td>
            <td>${row.smoking === 1 ? 'Yes' : 'No'}</td>
            <td><strong>${(row.cvd_risk_percentage || row.cvd_risk_probability * 100).toFixed(1)}%</strong></td>
            <td><span class="tag-badge ${tierClass}">${row.risk_tier}</span></td>
        `;
        tbody.appendChild(tr);
    });

    container.scrollIntoView({ behavior: 'smooth' });
}

/* ==================== 6. MODEL METRICS & CHART ==================== */
async function initModelMetrics() {
    document.getElementById('btn-refresh-metrics')?.addEventListener('click', fetchModelMetrics);
    await fetchModelMetrics();
}

async function fetchModelMetrics() {
    try {
        const response = await fetch('/api/v1/model-metrics');
        if (!response.ok) return;
        const data = await response.json();

        // Update KPI cards
        document.getElementById('metric-roc-auc').textContent = data.roc_auc.toFixed(3);
        document.getElementById('metric-accuracy').textContent = `${(data.accuracy * 100).toFixed(1)}%`;
        document.getElementById('metric-f1').textContent = data.f1_score.toFixed(3);
        document.getElementById('metric-records').textContent = data.dataset_size.toLocaleString();
        
        const headerStatus = document.getElementById('header-status-text');
        if (headerStatus) {
            headerStatus.textContent = `${data.model_name} (AUC: ${(data.roc_auc * 100).toFixed(1)}%)`;
        }

        // Render Feature Importance Chart
        renderFeatureChart(data.feature_importances);

        // Render Benchmark Table
        renderBenchmarkTable(data.all_model_benchmarks, data.model_name);

    } catch (e) {
        console.error("Failed to load metrics", e);
    }
}

function renderFeatureChart(importances) {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx) return;

    const labels = [];
    const values = [];

    for (const [_, item] of Object.entries(importances)) {
        labels.push(item.display_name);
        values.push(item.importance_pct);
    }

    if (window.featureChart) {
        window.featureChart.destroy();
    }

    window.featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.slice(0, 8),
            datasets: [{
                label: 'Relative Importance (%)',
                data: values.slice(0, 8),
                backgroundColor: [
                    '#3b82f6', '#06b6d4', '#10b981', '#f59e0b',
                    '#8b5cf6', '#ec4899', '#f97316', '#64748b'
                ],
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e2e8f0', font: { size: 11, family: 'Plus Jakarta Sans' } }
                }
            }
        }
    });
}

function renderBenchmarkTable(benchmarks, championName) {
    const tbody = document.getElementById('benchmark-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    for (const [name, res] of Object.entries(benchmarks)) {
        const isChampion = name === championName;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${name}</strong></td>
            <td>${res.cv_roc_auc_mean?.toFixed(4) || '--'}</td>
            <td>${(res.accuracy * 100).toFixed(2)}%</td>
            <td>${res.roc_auc.toFixed(4)}</td>
            <td>${res.precision.toFixed(4)}</td>
            <td>${res.recall.toFixed(4)}</td>
            <td>${res.f1_score.toFixed(4)}</td>
            <td>
                ${isChampion ? '<span class="tag-badge tag-champion"><i class="fa-solid fa-crown"></i> Champion</span>' : '<span class="tag-badge tag-moderate">Evaluated</span>'}
            </td>
        `;
        tbody.appendChild(tr);
    }
}

// Utility debounce
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
