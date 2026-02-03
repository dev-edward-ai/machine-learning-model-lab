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
    'SALES': '/samples/marketing_roi.csv'
};

const modelLabels = {
    'CRYPTO': 'Logistic Regression (Buy/Sell Signals)',
    'MEDICAL': 'K-Nearest Neighbors (Risk Scoring)',
    'CAR_PRICE': 'Decision Tree Regressor (Valuation)',
    'SALES': 'Random Forest Regressor (ROI)'
};

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
        const response = await fetch(csvPath);
        const csvText = await response.text();
        
        // Send to backend for training
        const formData = new FormData();
        const blob = new Blob([csvText], { type: 'text/csv' });
        const file = new File([blob], `${scenario}.csv`, { type: 'text/csv' });
        formData.append('file', file);
        
        const analysisResponse = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });
        
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
            'CRYPTO': 'crypto-page',
            'MEDICAL': 'medical-page',
            'CAR_PRICE': 'car-page',
            'SALES': 'sales-page'
        };
        
        showLoading(false);
        showPage(pageMap[scenario]);
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
});

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    showPage('selector-page');
});
