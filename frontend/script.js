/**
 * Business Insight Generator - Simplified Frontend
 * Auto-load sample CSVs and show real-world use cases
 * Production-ready with dynamic API URL detection
 */

// Detect if running locally or on production (Render)
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : window.location.origin;

let currentScenario = null;
let currentModel = null;

// CSV file mapping - relative paths work on both local and production
const csvMap = {
    'CRYPTO': '/samples/crypto_signals.csv',
    'MEDICAL': '/samples/heart_disease.csv',
    'CAR_PRICE': '/samples/used_car_prices.csv',
    'SALES': '/samples/marketing_roi.csv',
    // New scenarios
    'HOUSE':    '/samples/regression_housing.csv',
    'BANKNOTE': '/samples/banknote_authentication.csv',
    'CHURN':    '/samples/customer_churn.csv',
    'SPAM':     '/samples/sms_spam.csv',
    'SEGMENT':  '/samples/clustering_customers.csv',
    'STOCK':    '/samples/stock_sectors.csv'
};

const modelLabels = {
    'CRYPTO':   'Logistic Regression (Buy/Sell Signals)',
    'MEDICAL':  'K-Nearest Neighbors (Risk Scoring)',
    'CAR_PRICE':'Decision Tree Regressor (Valuation)',
    'SALES':    'Random Forest Regressor (ROI)',
    'HOUSE':    'Linear Regression (House Price)',
    'BANKNOTE': 'SVM Classifier (Authentication)',
    'CHURN':    'XGBoost Classifier (Churn)',
    'SPAM':     'Naive Bayes + TF-IDF (Spam)',
    'SEGMENT':  'K-Means (Segmentation)',
    'STOCK':    'PCA (Dimensionality Reduction)'
};

// Scenarios handled by the /analyze/new endpoint
const NEW_SCENARIOS = new Set(['HOUSE', 'BANKNOTE', 'CHURN', 'SPAM', 'SEGMENT', 'STOCK']);

// ============================================================================
// DATASET DOWNLOAD BAR
// ============================================================================
function injectDatasetBar(scenario, pageId) {
    document.querySelectorAll('.dataset-bar').forEach(el => el.remove());
    const csv = csvMap[scenario];
    if (!csv || !pageId) return;
    const page = document.getElementById(pageId);
    if (!page) return;
    const header = page.querySelector('.app-header');
    if (!header) return;
    const bar = document.createElement('div');
    bar.className = 'dataset-bar';
    const filename = csv.split('/').pop();
    const csvUrl = API_BASE_URL + csv;
    bar.innerHTML = `<i class="fas fa-database"></i> <strong>Training Dataset:</strong> ${filename} &nbsp;—&nbsp; <a href="${csvUrl}" target="_blank">Preview CSV</a> &nbsp;| <a href="${csvUrl}" download>⬇ Download</a>`;
    header.insertAdjacentElement('afterend', bar);
}

function updateModelChip(scenario, result) {
    const idMap = {
        'CRYPTO': 'crypto-model-chip',
        'MEDICAL': 'medical-model-chip',
        'CAR_PRICE': 'car-model-chip',
        'SALES': 'sales-model-chip'
    };
    const el = document.getElementById(idMap[scenario]);
    if (!el) return;
    const acc = result?.model_info?.accuracy;
    const accText = typeof acc === 'number' ? ` • Accuracy: ${(acc * 100).toFixed(1)}%` : '';
    el.textContent = `Model: ${modelLabels[scenario] || 'N/A'}${accText}`;
}

// ============================================================================
// PAGE NAVIGATION
// ============================================================================

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.toggle('active', show);
    }
}

function hidePage(pageId) {
    const page = document.getElementById(pageId);
    if (page) page.classList.remove('active');
}

function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    // Show target page
    const page = document.getElementById(pageId);
    if (page) page.classList.add('active');
    window.scrollTo(0, 0);
}

function goBack() {
    showPage('selector-page');
    currentScenario = null;
    currentModel = null;
}

// ============================================================================
// SCENARIO LOADING
// ============================================================================

async function loadScenario(scenario) {
    currentScenario = scenario;
    showLoading(true);
    
    try {
        // Load the CSV file
        const csvPath = csvMap[scenario];
        const response = await fetch(API_BASE_URL + csvPath);
        const csvText = await response.text();
        
        // Send to backend for training
        const formData = new FormData();
        const blob = new Blob([csvText], { type: 'text/csv' });
        const file = new File([blob], `${scenario}.csv`, { type: 'text/csv' });
        formData.append('file', file);

        let analysisResponse;
        if (NEW_SCENARIOS.has(scenario)) {
            // New scenarios: use dedicated /analyze/new endpoint
            formData.append('scenario_key', scenario);
            analysisResponse = await fetch(`${API_BASE_URL}/analyze/new`, {
                method: 'POST',
                body: formData
            });
        } else {
            analysisResponse = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                body: formData
            });
        }
        
        if (!analysisResponse.ok) {
            let detail = 'Failed to train model';
            try {
                const err = await analysisResponse.json();
                detail = err.detail || detail;
            } catch (_) {
                // ignore
            }
            throw new Error(detail);
        }
        
        const result = await analysisResponse.json();
        currentModel = result;
        updateModelChip(scenario, result);
        
        // Show appropriate page
        const pageMap = {
            'CRYPTO':   'crypto-page',
            'MEDICAL':  'medical-page',
            'CAR_PRICE':'car-page',
            'SALES':    'sales-page',
            'HOUSE':    'house-page',
            'BANKNOTE': 'banknote-page',
            'CHURN':    'churn-page',
            'SPAM':     'spam-page',
            'SEGMENT':  'segment-page',
            'STOCK':    'stock-page'
        };

        // Populate stock PCA panel from training response extras
        if (scenario === 'STOCK' && result.extras) {
            const ex = result.extras;
            // variance_explained is already in % (e.g. 68.5)
            document.getElementById('stock-var1').textContent =
                (ex.variance_explained?.[0] || ex.metrics?.explained_variance_pc1 || 0).toFixed(1) + '%';
            document.getElementById('stock-var2').textContent =
                (ex.variance_explained?.[1] || ex.metrics?.explained_variance_pc2 || 0).toFixed(1) + '%';
            const total = ex.metrics?.total_explained_variance ||
                ((ex.variance_explained?.[0] || 0) + (ex.variance_explained?.[1] || 0));
            document.getElementById('stock-var-total').textContent = total.toFixed(1) + '%';
            document.getElementById('stock-points').textContent =
                (ex.scatter_data?.length || 0).toLocaleString();
            const drivers = ex.top_drivers?.PC1 || [];
            document.getElementById('stock-drivers').textContent =
                drivers.slice(0, 5).join(', ') || '—';
            // Store feature_cols on the result so predictStock knows them
            if (ex.feature_cols) window._stockFeatureCols = ex.feature_cols;
            const resultBox = document.getElementById('stock-result');
            resultBox.className = 'result-display low-risk';
            resultBox.innerHTML = `
                <div class="result-icon"><i class="fas fa-check-circle"></i></div>
                <div class="result-text">PCA Complete</div>
                <div class="result-sub">${total.toFixed(1)}% variance retained in 2 components</div>
            `;
        }
        
        showLoading(false);
        showPage(pageMap[scenario]);
        injectDatasetBar(scenario, pageMap[scenario]);
    } catch (error) {
        showLoading(false);
        alert('Error loading scenario: ' + error.message);
    }
}

// ============================================================================
// REAL-WORLD USE CASES - PREDICTIONS
// ============================================================================

// CRYPTO TRADING
async function predictCrypto() {
    if (!currentModel || currentScenario !== 'CRYPTO') {
        alert('Model not loaded');
        return;
    }
    
    const open = parseFloat(document.getElementById('crypto-open').value);
    const close = parseFloat(document.getElementById('crypto-close').value);
    const volume = parseFloat(document.getElementById('crypto-volume').value);
    const rsi = parseFloat(document.getElementById('crypto-rsi').value);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/crypto`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ open, close, volume, rsi })
        });
        
        const prediction = await response.json();
        
        const resultBox = document.getElementById('crypto-result');
        const confidence = (prediction.confidence * 100).toFixed(1);
        
        let signalHTML = '';
        if (prediction.signal.includes('BUY')) {
            signalHTML = `
                <div class="result-icon"><i class="fas fa-arrow-trend-up"></i></div>
                <div class="result-text">STRONG BUY</div>
                <div class="result-sub">Market shows bullish momentum</div>
            `;
            resultBox.className = 'result-display buy';
        } else {
            signalHTML = `
                <div class="result-icon"><i class="fas fa-arrow-trend-down"></i></div>
                <div class="result-text">SELL</div>
                <div class="result-sub">Market shows bearish signals</div>
            `;
            resultBox.className = 'result-display sell';
        }
        
        resultBox.innerHTML = signalHTML;
        document.getElementById('crypto-confidence').textContent = confidence + '%';
        document.getElementById('crypto-recommendation').textContent = prediction.signal;
    } catch (error) {
        alert('Prediction error: ' + error.message);
    }
}

// MEDICAL DIAGNOSIS
async function predictMedical() {
    if (!currentModel || currentScenario !== 'MEDICAL') {
        alert('Model not loaded');
        return;
    }
    
    const age = parseFloat(document.getElementById('medical-age').value);
    const bmi = parseFloat(document.getElementById('medical-bmi').value);
    const bloodpressure = parseFloat(document.getElementById('medical-bp').value);
    const glucose = parseFloat(document.getElementById('medical-glucose').value);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/medical`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ age, bmi, bloodpressure, glucose })
        });
        
        const prediction = await response.json();
        const confidence = (prediction.confidence * 100).toFixed(1);
        
        const resultBox = document.getElementById('medical-result');
        let riskHTML = '';
        let riskLevel = prediction.risk_level;
        
        if (riskLevel.includes('Low')) {
            riskHTML = `
                <div class="result-icon"><i class="fas fa-check-circle"></i></div>
                <div class="result-text">Low Risk</div>
                <div class="result-sub">Patient appears healthy. Continue regular checkups.</div>
            `;
            resultBox.className = 'result-display low-risk';
        } else if (riskLevel.includes('Medium')) {
            riskHTML = `
                <div class="result-icon"><i class="fas fa-exclamation-triangle"></i></div>
                <div class="result-text">Medium Risk</div>
                <div class="result-sub">Monitor closely. Recommend lifestyle changes.</div>
            `;
            resultBox.className = 'result-display neutral';
        } else {
            riskHTML = `
                <div class="result-icon"><i class="fas fa-exclamation-circle"></i></div>
                <div class="result-text">High Risk</div>
                <div class="result-sub">Immediate medical consultation recommended.</div>
            `;
            resultBox.className = 'result-display high-risk';
        }
        
        resultBox.innerHTML = riskHTML;
        document.getElementById('medical-risk').textContent = riskLevel;
        document.getElementById('medical-confidence').textContent = confidence + '%';
    } catch (error) {
        alert('Prediction error: ' + error.message);
    }
}

// CAR PRICING
async function predictCar() {
    if (!currentModel || currentScenario !== 'CAR_PRICE') {
        alert('Model not loaded');
        return;
    }
    
    const year = parseFloat(document.getElementById('car-year').value);
    const mileage = parseFloat(document.getElementById('car-mileage').value);
    const horsepower = parseFloat(document.getElementById('car-hp').value);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/car_price`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ year, mileage, horsepower })
        });
        
        const prediction = await response.json();
        const price = Math.max(1000, Math.round(prediction.estimated_price));
        const confidence = (prediction.confidence * 100).toFixed(1);
        
        const resultBox = document.getElementById('car-result');
        resultBox.className = 'result-display price-display';
        resultBox.innerHTML = `
            <div class="price-tag">
                <span class="currency">$</span>
                <span class="price-value">${price.toLocaleString()}</span>
            </div>
            <div class="result-sub">Estimated Market Value</div>
        `;
        
        const priceRange = Math.round(price * 0.1);
        document.getElementById('car-range').textContent = 
            `$${(price - priceRange).toLocaleString()} - $${(price + priceRange).toLocaleString()}`;
        document.getElementById('car-confidence').textContent = confidence + '%';
    } catch (error) {
        alert('Prediction error: ' + error.message);
    }
}

// ============================================================================
// PREDICTIONS — NEW SCENARIOS
// ============================================================================

// HOUSE PRICE
async function predictHouse() {
    if (!currentModel || currentScenario !== 'HOUSE') { alert('Model not loaded'); return; }
    const square_footage = parseFloat(document.getElementById('house-sqft').value);
    const bedrooms       = parseFloat(document.getElementById('house-beds').value);
    const bathrooms      = parseFloat(document.getElementById('house-baths').value);
    const location_score = parseFloat(document.getElementById('house-loc').value);
    const house_age      = parseFloat(document.getElementById('house-age').value);
    try {
        const resp = await fetch(`${API_BASE_URL}/predict/house`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ square_footage, bedrooms, bathrooms, location_score, house_age })
        });
        const p = await resp.json();
        if (p.error) { alert(p.error); return; }
        const price = Math.round(p.predicted_price);
        const resultBox = document.getElementById('house-result');
        resultBox.className = 'result-display price-display';
        document.getElementById('house-price').textContent = price.toLocaleString();
        const metrics = currentModel.metrics || {};
        document.getElementById('house-r2').textContent  = metrics.r2 ? metrics.r2.toFixed(3) : '—';
        document.getElementById('house-rmse').textContent = metrics.rmse ? '$' + Math.round(metrics.rmse).toLocaleString() : '—';
    } catch(e) { alert('Prediction error: ' + e.message); }
}

// BANKNOTE AUTHENTICATION
async function predictBanknote() {
    if (!currentModel || currentScenario !== 'BANKNOTE') { alert('Model not loaded'); return; }
    const variance = parseFloat(document.getElementById('note-var').value);
    const skewness = parseFloat(document.getElementById('note-skew').value);
    const curtosis = parseFloat(document.getElementById('note-curt').value);
    const entropy  = parseFloat(document.getElementById('note-ent').value);
    try {
        const resp = await fetch(`${API_BASE_URL}/predict/banknote`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ variance, skewness, curtosis, entropy })
        });
        const p = await resp.json();
        if (p.error) { alert(p.error); return; }
        const resultBox = document.getElementById('banknote-result');
        if (p.authentic) {
            resultBox.className = 'result-display genuine';
            resultBox.innerHTML = `
                <div class="result-icon"><i class="fas fa-check-circle"></i></div>
                <div class="result-text">GENUINE</div>
                <div class="result-sub">High probability authentic banknote</div>`;
        } else {
            resultBox.className = 'result-display fake';
            resultBox.innerHTML = `
                <div class="result-icon"><i class="fas fa-times-circle"></i></div>
                <div class="result-text">COUNTERFEIT</div>
                <div class="result-sub">Signature matches forged banknote pattern</div>`;
        }
        document.getElementById('note-confidence').textContent = (p.confidence * 100).toFixed(1) + '%';
        const acc = currentModel.metrics?.accuracy;
        document.getElementById('note-accuracy').textContent = acc ? (acc * 100).toFixed(1) + '%' : '—';
    } catch(e) { alert('Prediction error: ' + e.message); }
}

// CUSTOMER CHURN
async function predictChurn() {
    if (!currentModel || currentScenario !== 'CHURN') { alert('Model not loaded'); return; }
    const tenure          = parseFloat(document.getElementById('churn-tenure').value);
    const monthly_charges = parseFloat(document.getElementById('churn-charges').value);
    const support_tickets = parseFloat(document.getElementById('churn-tickets').value);
    try {
        const resp = await fetch(`${API_BASE_URL}/predict/churn`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tenure, monthly_charges, support_tickets })
        });
        const p = await resp.json();
        if (p.error) { alert(p.error); return; }
        const resultBox = document.getElementById('churn-result');
        const isHigh = p.risk_level === 'High';
        resultBox.className = 'result-display ' + (isHigh ? 'churn-high' : 'churn-low');
        resultBox.innerHTML = `
            <div class="result-icon"><i class="fas fa-${isHigh ? 'exclamation-triangle' : 'check-circle'}"></i></div>
            <div class="result-text">${isHigh ? 'Likely to Churn' : 'Likely to Stay'}</div>
            <div class="result-sub">${isHigh ? 'Intervention recommended immediately' : 'Customer appears satisfied'}</div>`;
        document.getElementById('churn-prob').textContent = p.churn_probability.toFixed(1) + '%';
        document.getElementById('churn-risk').textContent = p.risk_level;
    } catch(e) { alert('Prediction error: ' + e.message); }
}

// SMS SPAM
async function predictSpam() {
    if (!currentModel || currentScenario !== 'SPAM') { alert('Model not loaded'); return; }
    const text = document.getElementById('spam-text').value.trim();
    if (!text) { alert('Please enter a message'); return; }
    try {
        const resp = await fetch(`${API_BASE_URL}/predict/spam`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const p = await resp.json();
        if (p.error) { alert(p.error); return; }
        const resultBox = document.getElementById('spam-result');
        resultBox.className = 'result-display ' + (p.is_spam ? 'is-spam' : 'is-ham');
        resultBox.innerHTML = `
            <div class="result-icon"><i class="fas fa-${p.is_spam ? 'skull-crossbones' : 'envelope-open-text'}"></i></div>
            <div class="result-text">${p.is_spam ? 'SPAM' : 'LEGITIMATE'}</div>
            <div class="result-sub">${p.is_spam ? 'This message matches spam patterns' : 'Message appears to be genuine'}</div>`;
        document.getElementById('spam-prob').textContent = (p.spam_probability * 100).toFixed(1) + '%';
        document.getElementById('spam-words').textContent =
            (p.top_spam_words || []).slice(0, 3).join(', ') || '—';
    } catch(e) { alert('Prediction error: ' + e.message); }
}

// STOCK PCA PROJECTOR
async function predictStock() {
    if (!currentModel || currentScenario !== 'STOCK') { alert('Model not loaded'); return; }
    const payload = {
        tech_score:    parseFloat(document.getElementById('stock-tech').value  || 0),
        value_score:   parseFloat(document.getElementById('stock-val').value   || 0),
        growth_score:  parseFloat(document.getElementById('stock-growth').value|| 0),
        pe_ratio:      parseFloat(document.getElementById('stock-pe').value    || 0),
        rsi:           parseFloat(document.getElementById('stock-rsi').value   || 0),
    };
    try {
        const resp = await fetch(`${API_BASE_URL}/predict/stock`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const p = await resp.json();
        if (p.error || p.detail) { alert(p.error || p.detail); return; }
        const resultBox = document.getElementById('stock-result');
        resultBox.className = 'result-display low-risk';
        resultBox.innerHTML = `
            <div class="result-icon"><i class="fas fa-crosshairs"></i></div>
            <div class="result-text">Projected</div>
            <div class="result-sub">PC1: <strong>${p.pc1}</strong> &nbsp;&nbsp; PC2: <strong>${p.pc2}</strong></div>`;
        document.getElementById('stock-proj-pc1').textContent = p.pc1;
        document.getElementById('stock-proj-pc2').textContent = p.pc2;
    } catch(e) { alert('Projection error: ' + e.message); }
}

// ============================================================================
// CUSTOMER SEGMENTATION
async function predictSegment() {
    if (!currentModel || currentScenario !== 'SEGMENT') { alert('Model not loaded'); return; }
    const annual_income   = parseFloat(document.getElementById('seg-income').value);
    const spending_score  = parseFloat(document.getElementById('seg-score').value);
    try {
        const resp = await fetch(`${API_BASE_URL}/predict/segment`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ annual_income, spending_score })
        });
        const p = await resp.json();
        if (p.error) { alert(p.error); return; }
        const resultBox = document.getElementById('segment-result');
        resultBox.className = 'result-display low-risk';
        resultBox.innerHTML = `
            <div class="result-icon"><i class="fas fa-users"></i></div>
            <div class="result-text">Cluster ${p.cluster}</div>
            <div class="result-sub">Customer assigned to segment ${p.cluster}</div>`;
        document.getElementById('seg-cluster').textContent = p.cluster;
        document.getElementById('seg-confidence').textContent =
            p.assignment_confidence ? (p.assignment_confidence * 100).toFixed(1) + '%' : '—';
    } catch(e) { alert('Prediction error: ' + e.message); }
}

// ============================================================================
// SALES FORECASTING
async function predictSales() {
    if (!currentModel || currentScenario !== 'SALES') {
        alert('Model not loaded');
        return;
    }
    
    const adspend = parseFloat(document.getElementById('sales-spend').value);
    const socialclicks = parseFloat(document.getElementById('sales-clicks').value);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict/sales`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ adspend, socialclicks })
        });
        
        const prediction = await response.json();
        const revenue = Math.max(1000, Math.round(prediction.predicted_revenue));
        const roi = prediction.roi_multiplier.toFixed(2);
        const profit = Math.round(revenue - adspend);
        
        const resultBox = document.getElementById('sales-result');
        resultBox.className = 'result-display price-display';
        resultBox.innerHTML = `
            <div class="price-tag">
                <span class="currency">$</span>
                <span class="price-value">${revenue.toLocaleString()}</span>
            </div>
            <div class="result-sub">Predicted Revenue</div>
        `;
        
        document.getElementById('sales-roi').textContent = roi;
        document.getElementById('sales-profit').textContent = 
            `$${Math.max(0, profit).toLocaleString()}`;
    } catch (error) {
        alert('Prediction error: ' + error.message);
    }
}

// ============================================================================
// SLIDER VALUE UPDATES
// ============================================================================

document.addEventListener('input', (e) => {
    if (e.target.id === 'crypto-rsi') {
        document.getElementById('crypto-rsi-value').textContent = e.target.value;
    }
    if (e.target.id === 'medical-age') {
        document.getElementById('medical-age-value').textContent = e.target.value;
    }
    if (e.target.id === 'medical-bmi') {
        document.getElementById('medical-bmi-value').textContent = e.target.value;
    }
    if (e.target.id === 'medical-glucose') {
        document.getElementById('medical-glucose-value').textContent = e.target.value;
    }
    if (e.target.id === 'car-year') {
        document.getElementById('car-year-value').textContent = e.target.value;
    }
    if (e.target.id === 'car-hp') {
        document.getElementById('car-hp-value').textContent = e.target.value + ' HP';
    }
    if (e.target.id === 'sales-spend') {
        const val = parseFloat(e.target.value);
        document.getElementById('sales-spend-value').textContent = '$' + val.toLocaleString();
    }
    if (e.target.id === 'sales-clicks') {
        const val = parseFloat(e.target.value);
        document.getElementById('sales-clicks-value').textContent = val.toLocaleString();
    }
    // New scenario sliders
    if (e.target.id === 'house-sqft')    document.getElementById('house-sqft-value').textContent = parseInt(e.target.value).toLocaleString();
    if (e.target.id === 'house-beds')    document.getElementById('house-beds-value').textContent = e.target.value;
    if (e.target.id === 'house-baths')   document.getElementById('house-baths-value').textContent = e.target.value;
    if (e.target.id === 'house-loc')     document.getElementById('house-loc-value').textContent = e.target.value;
    if (e.target.id === 'house-age')     document.getElementById('house-age-value').textContent = e.target.value;
    if (e.target.id === 'note-var')      document.getElementById('note-var-value').textContent = parseFloat(e.target.value).toFixed(1);
    if (e.target.id === 'note-skew')     document.getElementById('note-skew-value').textContent = parseFloat(e.target.value).toFixed(1);
    if (e.target.id === 'note-curt')     document.getElementById('note-curt-value').textContent = parseFloat(e.target.value).toFixed(1);
    if (e.target.id === 'note-ent')      document.getElementById('note-ent-value').textContent = parseFloat(e.target.value).toFixed(1);
    if (e.target.id === 'churn-tenure')  document.getElementById('churn-tenure-value').textContent = e.target.value;
    if (e.target.id === 'churn-tickets') document.getElementById('churn-tickets-value').textContent = e.target.value;
    if (e.target.id === 'seg-income')    document.getElementById('seg-income-value').textContent = parseInt(e.target.value).toLocaleString();
    if (e.target.id === 'seg-score')     document.getElementById('seg-score-value').textContent = e.target.value;
    // Stock sliders
    if (e.target.id === 'stock-tech')   document.getElementById('stock-tech-val').textContent = parseFloat(e.target.value).toFixed(1);
    if (e.target.id === 'stock-val')    document.getElementById('stock-val-val').textContent = parseFloat(e.target.value).toFixed(1);
    if (e.target.id === 'stock-growth') document.getElementById('stock-growth-val').textContent = parseFloat(e.target.value).toFixed(1);
    if (e.target.id === 'stock-pe')     document.getElementById('stock-pe-val').textContent = e.target.value;
    if (e.target.id === 'stock-rsi')    document.getElementById('stock-rsi-val').textContent = e.target.value;
});

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    showPage('selector-page');
});
