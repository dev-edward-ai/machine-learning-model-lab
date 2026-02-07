// ===================================
// AutoML Intelligence Platform v2.0
// Interactive Microsites & ResultViewFactory
// ===================================

const API_BASE_URL = 'http://localhost:8000';

// ===================================
// Result View Factory
// Dynamically renders UI based on detected scenario
// ===================================

class ResultViewFactory {
    static createComponent(analysisResult) {
        const uiConfig = analysisResult.ui_config;
        const componentType = uiConfig.component_type;
        
        switch (componentType) {
            case 'LoanOfficerDashboard':
                return new LoanOfficerDashboard(analysisResult);
            case 'TradingTerminal':
                return new TradingTerminal(analysisResult);
            case 'ClusteringVisualization':
                return new ClusteringVisualization(analysisResult);
            case 'PricingCalculator':
                return new PricingCalculator(analysisResult);
            case 'SpamDetector':
                return new SpamDetector(analysisResult);
            case 'MedicalDashboard':
                return new MedicalDashboard(analysisResult);
            default:
                return new GeneralResults(analysisResult);
        }
    }
}

// ===================================
// Base Microsite Class
// ===================================

class BaseMicrosite {
    constructor(analysisResult) {
        this.data = analysisResult;
        this.container = null;
    }
    
    render(containerId) {
        this.container = document.getElementById(containerId);
        this.container.innerHTML = this.generateHTML();
        this.attachEventListeners();
        this.initializeComponent();
    }
    
    generateHTML() {
        return '<div class="microsite-placeholder">Base microsite</div>';
    }
    
    attachEventListeners() {
        // Override in subclasses
    }
    
    initializeComponent() {
        // Override in subclasses
    }
}

// ===================================
// Loan Officer Dashboard
// Interactive loan approval interface
// ===================================

class LoanOfficerDashboard extends BaseMicrosite {
    generateHTML() {
        const inputFields = this.data.ui_config.input_fields;
        
        return `
            <div class="loan-dashboard">
                <div class="dashboard-header">
                    <div class="header-icon">🏦</div>
                    <h2>Loan Officer Dashboard</h2>
                    <p>Enter applicant details for instant approval decision</p>
                </div>
                
                <div class="dashboard-content">
                    <div class="input-panel">
                        <form id="loan-form" class="loan-form">
                            ${inputFields.map(field => `
                                <div class="input-group">
                                    <label for="${field.name}">${field.label}</label>
                                    <input 
                                        type="${field.type}" 
                                        id="${field.name}" 
                                        name="${field.name}"
                                        min="${field.min || ''}"
                                        max="${field.max || ''}"
                                        step="${field.step || 'any'}"
                                        required
                                        class="form-input"
                                    />
                                </div>
                            `).join('')}
                            
                            <button type="submit" class="analyze-btn" id="loan-submit">
                                <span class="btn-icon">🔍</span>
                                Analyze Application
                            </button>
                        </form>
                    </div>
                    
                    <div class="result-panel">
                        <div id="loan-result" class="loan-result hidden">
                            <div class="result-header">
                                <div class="result-status" id="approval-status"></div>
                                <div class="confidence-score" id="confidence-score"></div>
                            </div>
                            
                            <div class="result-details">
                                <div class="decision-factors" id="decision-factors"></div>
                                <div class="risk-assessment" id="risk-assessment"></div>
                            </div>
                        </div>
                        
                        <div class="sample-cases">
                            <h4>Sample Cases</h4>
                            ${this.generateSampleCases()}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    generateSampleCases() {
        const samples = this.data.ui_config.sample_data;
        return samples.map((sample, index) => `
            <div class="sample-case" data-sample='${JSON.stringify(sample)}'>
                <span class="sample-label">Case ${index + 1}</span>
                <span class="sample-values">
                    Income: $${sample.income || 'N/A'} | 
                    Credit: ${sample.credit_score || 'N/A'}
                </span>
            </div>
        `).join('');
    }
    
    attachEventListeners() {
        const form = document.getElementById('loan-form');
        const sampleCases = document.querySelectorAll('.sample-case');
        
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeLoanApplication();
        });
        
        // Sample case click handlers
        sampleCases.forEach(caseEl => {
            caseEl.addEventListener('click', () => {
                const sampleData = JSON.parse(caseEl.dataset.sample);
                this.populateForm(sampleData);
            });
        });
        
        // Real-time validation
        form.addEventListener('input', () => {
            this.validateForm();
        });
    }
    
    populateForm(data) {
        Object.keys(data).forEach(key => {
            const input = document.getElementById(key);
            if (input) {
                input.value = data[key];
            }
        });
        this.validateForm();
    }
    
    validateForm() {
        const form = document.getElementById('loan-form');
        const submitBtn = document.getElementById('loan-submit');
        const isValid = form.checkValidity();
        
        submitBtn.disabled = !isValid;
        submitBtn.classList.toggle('disabled', !isValid);
    }
    
    async analyzeLoanApplication() {
        const form = document.getElementById('loan-form');
        const formData = new FormData(form);
        const resultContainer = document.getElementById('loan-result');
        const submitBtn = document.getElementById('loan-submit');
        
        // Show loading state
        submitBtn.innerHTML = '<span class="spinner"></span>Analyzing...';
        submitBtn.disabled = true;
        
        try {
            // Simulate API call (replace with actual prediction endpoint)
            const response = await this.simulateLoanPrediction(formData);
            
            this.displayLoanResult(response);
            resultContainer.classList.remove('hidden');
            
        } catch (error) {
            console.error('Loan analysis failed:', error);
            this.showError('Analysis failed. Please try again.');
        } finally {
            submitBtn.innerHTML = '<span class="btn-icon">🔍</span>Analyze Application';
            submitBtn.disabled = false;
        }
    }
    
    async simulateLoanPrediction(formData) {
        // Simulate prediction logic based on form data
        const income = parseFloat(formData.get('income'));
        const creditScore = parseFloat(formData.get('credit_score'));
        const debtToIncome = parseFloat(formData.get('debt_to_income'));
        const employmentYears = parseFloat(formData.get('employment_years'));
        
        // Simple rule-based simulation
        let score = 50;
        if (income > 50000) score += 15;
        if (income > 80000) score += 10;
        if (creditScore > 700) score += 20;
        if (creditScore > 750) score += 10;
        if (debtToIncome < 0.3) score += 15;
        if (employmentYears > 3) score += 10;
        
        const approved = score > 70;
        const confidence = Math.min(score, 95);
        
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        return {
            approved,
            confidence,
            score,
            factors: this.generateDecisionFactors(income, creditScore, debtToIncome, employmentYears),
            risk_level: approved ? 'Low' : 'High'
        };
    }
    
    generateDecisionFactors(income, creditScore, debtToIncome, employmentYears) {
        const factors = [];
        
        if (creditScore > 750) factors.push('✅ Excellent credit score');
        else if (creditScore > 700) factors.push('✅ Good credit score');
        else factors.push('⚠️ Credit score needs improvement');
        
        if (income > 80000) factors.push('✅ Strong income level');
        else if (income > 50000) factors.push('✅ Adequate income');
        else factors.push('⚠️ Income below threshold');
        
        if (debtToIncome < 0.3) factors.push('✅ Low debt-to-income ratio');
        else factors.push('⚠️ High debt burden');
        
        if (employmentYears > 3) factors.push('✅ Stable employment history');
        else factors.push('⚠️ Limited employment history');
        
        return factors;
    }
    
    displayLoanResult(result) {
        const statusEl = document.getElementById('approval-status');
        const confidenceEl = document.getElementById('confidence-score');
        const factorsEl = document.getElementById('decision-factors');
        const riskEl = document.getElementById('risk-assessment');
        
        // Status display
        statusEl.className = `result-status ${result.approved ? 'approved' : 'denied'}`;
        statusEl.innerHTML = `
            <div class="status-icon">${result.approved ? '✅' : '❌'}</div>
            <div class="status-text">
                <h3>${result.approved ? 'APPROVED' : 'DENIED'}</h3>
                <p>Application ${result.approved ? 'approved' : 'rejected'}</p>
            </div>
        `;
        
        // Confidence score
        confidenceEl.innerHTML = `
            <div class="confidence-label">Confidence</div>
            <div class="confidence-value">${result.confidence.toFixed(1)}%</div>
        `;
        
        // Decision factors
        factorsEl.innerHTML = `
            <h4>Decision Factors</h4>
            <ul class="factors-list">
                ${result.factors.map(factor => `<li>${factor}</li>`).join('')}
            </ul>
        `;
        
        // Risk assessment
        riskEl.innerHTML = `
            <h4>Risk Assessment</h4>
            <div class="risk-indicator ${result.risk_level.toLowerCase()}">
                ${result.risk_level} Risk
            </div>
        `;
    }
    
    showError(message) {
        const resultContainer = document.getElementById('loan-result');
        resultContainer.innerHTML = `
            <div class="error-message">
                <span class="error-icon">⚠️</span>
                ${message}
            </div>
        `;
        resultContainer.classList.remove('hidden');
    }
}

// ===================================
// Trading Terminal
// Interactive crypto trading interface
// ===================================

class TradingTerminal extends BaseMicrosite {
    constructor(analysisResult) {
        super(analysisResult);
        this.chart = null;
        this.liveData = this.generateLiveData();
        this.currentSignal = 'NEUTRAL';
    }
    
    generateHTML() {
        return `
            <div class="trading-terminal">
                <div class="terminal-header">
                    <div class="header-icon">💰</div>
                    <h2>Crypto Trading Terminal</h2>
                    <div class="live-indicator">
                        <span class="pulse-dot"></span>
                        LIVE MARKET
                    </div>
                </div>
                
                <div class="terminal-content">
                    <div class="signal-panel">
                        <div class="current-signal" id="trading-signal">
                            <div class="signal-status">ANALYZING</div>
                            <div class="signal-strength">--</div>
                        </div>
                        
                        <div class="signal-metrics">
                            <div class="metric">
                                <label>RSI</label>
                                <span id="rsi-value">--</span>
                            </div>
                            <div class="metric">
                                <label>MACD</label>
                                <span id="macd-value">--</span>
                            </div>
                            <div class="metric">
                                <label>Volume</label>
                                <span id="volume-value">--</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="chart-panel">
                        <div class="chart-controls">
                            <button class="timeframe-btn active" data-timeframe="1H">1H</button>
                            <button class="timeframe-btn" data-timeframe="4H">4H</button>
                            <button class="timeframe-btn" data-timeframe="1D">1D</button>
                        </div>
                        <canvas id="trading-chart" class="trading-chart"></canvas>
                    </div>
                    
                    <div class="indicators-panel">
                        <div class="indicator-group">
                            <h4>Technical Indicators</h4>
                            <div class="indicators-grid" id="indicators-grid">
                                <!-- Dynamic indicators will be populated here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    generateLiveData() {
        // Generate realistic crypto trading data
        const data = [];
        let price = 45000 + Math.random() * 10000;
        let volume = 1000000;
        
        for (let i = 0; i < 50; i++) {
            price += (Math.random() - 0.5) * 500;
            volume += (Math.random() - 0.5) * 100000;
            
            const timestamp = Date.now() - (50 - i) * 60000; // 1 minute intervals
            
            data.push({
                timestamp,
                price: price + (Math.random() - 0.5) * 100,
                high: price + Math.random() * 200,
                low: price - Math.random() * 200,
                volume,
                rsi: 30 + Math.random() * 40,
                macd: (Math.random() - 0.5) * 1000,
                moving_average_7: price + (Math.random() - 0.5) * 100,
                moving_average_30: price + (Math.random() - 0.5) * 200
            });
        }
        
        return data;
    }
    
    attachEventListeners() {
        // Timeframe buttons
        document.querySelectorAll('.timeframe-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.timeframe-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.updateChart(btn.dataset.timeframe);
            });
        });
        
        // Start live updates
        this.startLiveUpdates();
    }
    
    initializeComponent() {
        this.initializeChart();
        this.updateSignalPanel();
        this.updateIndicators();
    }
    
    initializeChart() {
        const canvas = document.getElementById('trading-chart');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = 400 * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        canvas.style.height = '400px';
        
        this.renderChart(ctx);
    }
    
    renderChart(ctx) {
        const width = ctx.canvas.width / window.devicePixelRatio;
        const height = ctx.canvas.height / window.devicePixelRatio;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid
        this.drawGrid(ctx, width, height);
        
        // Draw price line
        this.drawPriceLine(ctx, width, height);
        
        // Draw indicators
        this.drawIndicators(ctx, width, height);
    }
    
    drawGrid(ctx, width, height) {
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.1)';
        ctx.lineWidth = 1;
        
        // Vertical lines
        for (let i = 0; i <= 10; i++) {
            const x = (width / 10) * i;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        
        // Horizontal lines
        for (let i = 0; i <= 8; i++) {
            const y = (height / 8) * i;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
    }
    
    drawPriceLine(ctx, width, height) {
        const data = this.liveData.slice(-20); // Last 20 data points
        const prices = data.map(d => d.price);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const priceRange = maxPrice - minPrice;
        
        ctx.strokeStyle = '#10B981';
        ctx.lineWidth = 3;
        ctx.beginPath();
        
        data.forEach((point, index) => {
            const x = (width / (data.length - 1)) * index;
            const y = height - ((point.price - minPrice) / priceRange) * height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw current price point
        if (data.length > 0) {
            const lastPoint = data[data.length - 1];
            const x = width - 10;
            const y = height - ((lastPoint.price - minPrice) / priceRange) * height;
            
            ctx.fillStyle = '#10B981';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    drawIndicators(ctx, width, height) {
        // Draw RSI indicator (simplified)
        const rsi = this.liveData[this.liveData.length - 1].rsi;
        const rsiHeight = 30;
        
        ctx.fillStyle = 'rgba(139, 92, 246, 0.1)';
        ctx.fillRect(0, height - rsiHeight, width, rsiHeight);
        
        ctx.fillStyle = rsi > 70 ? '#EF4444' : rsi < 30 ? '#10B981' : '#8B5CF6';
        const rsiWidth = (rsi / 100) * width;
        ctx.fillRect(0, height - rsiHeight, rsiWidth, rsiHeight);
    }
    
    updateSignalPanel() {
        const latestData = this.liveData[this.liveData.length - 1];
        
        // Determine signal based on indicators
        const signal = this.calculateTradingSignal(latestData);
        
        const signalEl = document.getElementById('trading-signal');
        signalEl.className = `current-signal ${signal.status.toLowerCase()}`;
        signalEl.innerHTML = `
            <div class="signal-status">${signal.status}</div>
            <div class="signal-strength">${signal.strength}%</div>
        `;
        
        // Update metrics
        document.getElementById('rsi-value').textContent = latestData.rsi.toFixed(1);
        document.getElementById('macd-value').textContent = latestData.macd.toFixed(0);
        document.getElementById('volume-value').textContent = this.formatVolume(latestData.volume);
    }
    
    calculateTradingSignal(data) {
        let score = 0;
        
        // RSI analysis
        if (data.rsi > 70) score -= 20; // Overbought
        else if (data.rsi < 30) score += 20; // Oversold
        
        // MACD analysis
        if (data.macd > 0) score += 15;
        else score -= 15;
        
        // Moving average analysis
        if (data.moving_average_7 > data.moving_average_30) score += 15;
        else score -= 15;
        
        // Price momentum
        if (this.liveData.length > 1) {
            const prevPrice = this.liveData[this.liveData.length - 2].price;
            if (data.price > prevPrice) score += 10;
            else score -= 10;
        }
        
        const strength = Math.min(Math.abs(score), 95);
        
        if (score > 40) return { status: 'BUY', strength };
        else if (score < -40) return { status: 'SELL', strength };
        else return { status: 'HOLD', strength };
    }
    
    formatVolume(volume) {
        if (volume > 1000000) {
            return (volume / 1000000).toFixed(1) + 'M';
        } else if (volume > 1000) {
            return (volume / 1000).toFixed(1) + 'K';
        }
        return volume.toString();
    }
    
    updateIndicators() {
        const indicatorsGrid = document.getElementById('indicators-grid');
        const latestData = this.liveData[this.liveData.length - 1];
        
        const indicators = [
            {
                name: 'RSI (14)',
                value: latestData.rsi.toFixed(1),
                status: latestData.rsi > 70 ? 'overbought' : latestData.rsi < 30 ? 'oversold' : 'neutral'
            },
            {
                name: 'MACD',
                value: latestData.macd.toFixed(0),
                status: latestData.macd > 0 ? 'bullish' : 'bearish'
            },
            {
                name: 'MA (7/30)',
                value: latestData.moving_average_7 > latestData.moving_average_30 ? 'Golden Cross' : 'Death Cross',
                status: latestData.moving_average_7 > latestData.moving_average_30 ? 'bullish' : 'bearish'
            }
        ];
        
        indicatorsGrid.innerHTML = indicators.map(indicator => `
            <div class="indicator-item ${indicator.status}">
                <div class="indicator-name">${indicator.name}</div>
                <div class="indicator-value">${indicator.value}</div>
            </div>
        `).join('');
    }
    
    startLiveUpdates() {
        setInterval(() => {
            // Simulate new data point
            const lastData = this.liveData[this.liveData.length - 1];
            const newData = {
                timestamp: Date.now(),
                price: lastData.price + (Math.random() - 0.5) * 200,
                volume: lastData.volume + (Math.random() - 0.5) * 50000,
                rsi: Math.max(0, Math.min(100, lastData.rsi + (Math.random() - 0.5) * 5)),
                macd: lastData.macd + (Math.random() - 0.5) * 100,
                moving_average_7: lastData.moving_average_7 + (Math.random() - 0.5) * 50,
                moving_average_30: lastData.moving_average_30 + (Math.random() - 0.5) * 30
            };
            
            this.liveData.push(newData);
            
            // Keep only last 50 points
            if (this.liveData.length > 50) {
                this.liveData.shift();
            }
            
            // Update UI
            this.updateSignalPanel();
            this.updateIndicators();
            
            // Redraw chart
            const canvas = document.getElementById('trading-chart');
            const ctx = canvas.getContext('2d');
            this.renderChart(ctx);
            
        }, 2000); // Update every 2 seconds
    }
    
    updateChart(timeframe) {
        console.log(`Switching to ${timeframe} timeframe`);
        // In a real implementation, this would fetch data for the selected timeframe
        this.initializeChart();
    }
}

// ===================================
// General Results (Fallback)
// ===================================

class GeneralResults extends BaseMicrosite {
    generateHTML() {
        const model = this.data.tournament_results.recommended_model;
        const topModels = this.data.tournament_results.top_models;
        
        return `
            <div class="general-results">
                <div class="results-header">
                    <div class="header-icon">🤖</div>
                    <h2>ML Analysis Results</h2>
                    <p>Top performing models for your dataset</p>
                </div>
                
                <div class="model-tournament">
                    <h3>Model Tournament Results</h3>
                    <div class="tournament-chart" id="tournament-chart">
                        ${this.generateTournamentChart(topModels)}
                    </div>
                </div>
                
                <div class="recommended-model">
                    <h3>Recommended Model</h3>
                    <div class="model-card">
                        <div class="model-name">${model.name}</div>
                        <div class="model-score">${model.score.toFixed(1)}%</div>
                        <div class="model-explanation">
                            ${model.explanation?.how_it_works || 'No explanation available'}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    generateTournamentChart(models) {
        return models.map((model, index) => `
            <div class="model-bar" style="--score: ${model.score}%; --delay: ${index * 0.1}s">
                <div class="model-info">
                    <span class="model-name">${model.name}</span>
                    <span class="model-score">${model.score.toFixed(1)}%</span>
                </div>
                <div class="score-bar"></div>
            </div>
        `).join('');
    }
}

// Export the main factory
window.ResultViewFactory = ResultViewFactory;