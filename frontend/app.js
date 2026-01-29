// ===================================

// AutoML Intelligence Platform

// Premium Frontend with Animations

// ===================================



const API_BASE_URL = 'http://localhost:8000';



// ===================================

// Particle System

// ===================================



class ParticleSystem {

    constructor(canvas) {

        this.canvas = canvas;

        this.ctx = canvas.getContext('2d');

        this.particles = [];

        this.mouseX = 0;

        this.mouseY = 0;



        this.resize();

        this.init();

        this.animate();



        window.addEventListener('resize', () => this.resize());

        document.addEventListener('mousemove', (e) => {

            this.mouseX = e.clientX;

            this.mouseY = e.clientY;

        });

    }



    resize() {

        this.canvas.width = window.innerWidth;

        this.canvas.height = document.documentElement.scrollHeight;

    }



    init() {

        const particleCount = Math.min(150, Math.floor((this.canvas.width * this.canvas.height) / 15000));

        for (let i = 0; i < particleCount; i++) {

            this.particles.push({

                x: Math.random() * this.canvas.width,

                y: Math.random() * this.canvas.height,

                vx: (Math.random() - 0.5) * 0.5,

                vy: (Math.random() - 0.5) * 0.5,

                radius: Math.random() * 2 + 1,

                opacity: Math.random() * 0.5 + 0.2

            });

        }

    }



    animate() {

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);



        this.particles.forEach((particle, i) => {

            // Update position

            particle.x += particle.vx;

            particle.y += particle.vy;



            // Mouse intearaction

            const dx = this.mouseX - particle.x;

            const dy = this.mouseY - particle.y;

            const distance = Math.sqrt(dx * dx + dy * dy);



            if (distance < 100) {

                const foarce = (100 - distance) / 100;

                particle.vx -= (dx / distance) * foarce * 0.1;

                particle.vy -= (dy / distance) * foarce * 0.1;

            }



            // Wrp round screen

            if (particle.x < 0) particle.x = this.canvas.width;

            if (particle.x > this.canvas.width) particle.x = 0;

            if (particle.y < 0) particle.y = this.canvas.height;

            if (particle.y > this.canvas.height) particle.y = 0;



            // Drw particle

            this.ctx.beginPath();

            this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);

            this.ctx.fillStyle = `rgb(139, 92, 246, ${particle.opacity})`;

            this.ctx.fill();



            // Drw conneactions

            this.particles.slice(i + 1).forEach(particle2 => {

                const dx = particle.x - particle2.x;

                const dy = particle.y - particle2.y;

                const distance = Math.sqrt(dx * dx + dy * dy);



                if (distance < 100) {

                    this.ctx.beginPath();

                    this.ctx.moveTo(particle.x, particle.y);

                    this.ctx.lineTo(particle2.x, particle2.y);

                    this.ctx.strokeStyle = `rgb(139, 92, 246, ${0.15 * (1 - distance / 100)})`;

                    this.ctx.lineWidth = 1;

                    this.ctx.stroke();

                }

            });

        });



        requestAnimationFrame(() => this.animate());

    }

}



// ===================================

// Typing Aanimation

// ===================================



class TypingEffect {

    constructor(element, texts, typingSpeed = 80, deletingSpeed = 50, pauseDuration = 2000) {

        this.element = element;

        this.texts = texts;

        this.typingSpeed = typingSpeed;

        this.deletingSpeed = deletingSpeed;

        this.pauseDuration = pauseDuration;

        this.currentTextIndex = 0;

        this.currentCharIndex = 0;

        this.isDeleting = false;



        this.type();

    }



    type() {

        const currentText = this.texts[this.currentTextIndex];



        if (this.isDeleting) {

            this.element.textContent = currentText.substring(0, this.currentCharIndex - 1);

            this.currentCharIndex--;

        } else {

            this.element.textContent = currentText.substring(0, this.currentCharIndex + 1);

            this.currentCharIndex++;

        }



        // Add cursor

        this.element.innerHTML += '<span class="typing-cursor">|</span>';



        let timeout = this.isDeleting ? this.deletingSpeed : this.typingSpeed;



        if (!this.isDeleting && this.currentCharIndex === currentText.length) {

            timeout = this.pauseDuration;

            this.isDeleting = true;

        } else if (this.isDeleting && this.currentCharIndex === 0) {

            this.isDeleting = false;

            this.currentTextIndex = (this.currentTextIndex + 1) % this.texts.length;

            timeout = 500;

        }



        setTimeout(() => this.type(), timeout);

    }

}



// ===================================

// Smooth Scroll

// ===================================



document.querySelectorAll('[href^="#"]').forEach(anchor => {

    anchor.addEventListener('click', function (e) {

        e.preventDefault();

        const target = document.querySelector(this.getAttribute('href'));

        if (target) {

            target.scrollIntoView({

                behavior: 'smooth',

                block: 'start'

            });

        }

    });

});



// ===================================

// File Upload Handling

// ===================================



const fileInput = document.getElementById('fileInput');

const dropZone = document.getElementById('dropZone');

const uploadZoneContent = document.getElementById('uploadZoneContent');

const fileInfo = document.getElementById('fileInfo');

const fileName = document.getElementById('fileName');

const fileSize = document.getElementById('fileSize');

const fileRemove = document.getElementById('fileRemove');

const uploadProgress = document.getElementById('uploadProgress');

const progressFill = document.getElementById('progressFill');

const progressText = document.getElementById('progressText');



let selectedFile = null;



// Click to browse

dropZone.addEventListener('click', (e) => {

    // Don't trigger file input if clicking on remove button

    if (fileRemove && (e.target === fileRemove || fileRemove.contains(e.target))) {

        return;

    }

    fileInput.click();

});



// File selection

fileInput.addEventListener('change', handleFileSelect);



function handleFileSelect(e) {

    const files = e.target.files || e.dataTransfer?.files;

    if (files && files.length > 0) {

        selectedFile = files[0];

        displayFileInfo(selectedFile);

    }

}



function displayFileInfo(file) {

    fileName.textContent = file.name;

    fileSize.textContent = formatFileSize(file.size);



    uploadZoneContent.classList.add('hidden');

    fileInfo.classList.remove('hidden');

    dropZone.classList.add('has-file');

}



function formatFileSize(bytes) {

    if (bytes === 0) return '0 Bytes';

    const k = 1024;

    const sizes = ['Bytes', 'KB', 'MB', 'GB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];

}



// Remove file

fileRemove.addEventListener('click', (e) => {

    e.stopPropagation();

    resetFileUpload();

});



function resetFileUpload() {

    selectedFile = null;

    fileInput.value = '';

    uploadZoneContent.classList.remove('hidden');

    fileInfo.classList.add('hidden');

    dropZone.classList.remove('has-file');

}



// Drag and drop

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



    const files = e.dataTransfer.files;

    if (files.length > 0 && files[0].name.endsWith('.csv')) {

        // Create a DataTransfer object to properly set files
        const dataTransfer = new DataTransfer();

        dataTransfer.items.add(files[0]);

        fileInput.files = dataTransfer.files;

        handleFileSelect({ target: { files: dataTransfer.files } });

    } else {

        showNotification('⚠️ Please upload a CSV file', 'warning');

    }

});



// ===================================

// Form Submission

// ===================================



const uploadForm = document.getElementById('uploadForm');

const analyzeBtn = document.getElementById('analyzeBtn');

const statusIndicator = document.getElementById('statusIndicator');

const statusText = document.getElementById('statusText');

const resultsSeaction = document.getElementById('results');

const resultsContent = document.getElementById('resultsContent');

const resetBtn = document.getElementById('resetBtn');



uploadForm.addEventListener('submit', async (e) => {

    e.preventDefault();



    if (!selectedFile) {

        showNotification('⚠️ Please select  CSV file first', 'warning');

        return;

    }



    // Show loading state

    analyzeBtn.disabled = true;

    analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing...';

    if (statusIndicator) statusIndicator.classList.remove('hidden');

    statusText.textContent = 'Processing your dataset...';

    uploadProgress.classList.remove('hidden');

    progressFill.style.width = '20%';



    try {

        // Prepre form data

        const formData = new FormData();

        formData.append('file', selectedFile);



        const targetColumn = document.getElementById('targetColumn').value.trim();

        if (targetColumn) {

            formData.append('target_col', targetColumn);

        }



        const businessObjective = document.getElementById('businessObjective').value;

        if (businessObjective) {

            formData.append('business_objective', businessObjective);

            formData.append('user_intent', businessObjective);

        }



        // Update progress

        progressFill.style.width = '40%';

        statusText.textContent = 'Running model tournament...';



        // Make API request

        const response = await fetch(`${API_BASE_URL}/smart-dispatch`, {

            method: 'POST',

            body: formData

        });



        progressFill.style.width = '70%';



        if (!response.ok) {

            let errMsg = 'Analysis failed';

            try {

                const errorData = await response.json();

                const detail = errorData.detail;

                errMsg = (typeof detail === 'string' ? detail : detail?.error || detail?.msg) || errMsg;

            } catch (_) {

                errMsg = response.statusText || errMsg;

            }

            throw new Error(errMsg);

        }



        const data = await response.json();



        // Comaplete progress

        progressFill.style.width = '100%';

        statusText.textContent = 'Analysis complete!';



        // Display results (normalize smart-dispatch response for displayResults)

        const normalized = normalizeSmartDispatchResponse(data);

        setTimeout(() => {

            displayResults(normalized);

            uploadProgress.classList.add('hidden');

            if (statusIndicator) statusIndicator.classList.add('hidden');



            // Scroll to results

            resultsSeaction.scrollIntoView({ behavior: 'smooth' });



            showNotification('✅ Analysis completed successfully!', 'success');



            // Show demo modal fter  brief delay (if session_id exists)

            if (data.session_id && data.scenario) {

                setTimeout(() => {

                    showDemoModal(data);

                }, 800);

            }

        }, 500);



    } catch (error) {

        console.error('Error:', error);

        uploadProgress.classList.add('hidden');

        if (statusIndicator) statusIndicator.classList.add('hidden');



        showNotification(`❌ ${error.message}`, 'error');

        if (statusIndicator) statusIndicator.classList.add('hidden');



        resultsContent.innerHTML = `

<div class="error-crd" style = "

                background: linear - gradient(135deg, rgb(239, 68, 68, 0.1) 0 %, rgb(220, 38, 38, 0.05) 100 %);

border: 2px solid rgb(239, 68, 68, 0.3);

border-radius: 20px;

padding: 2rem;

text-align: center;

">

    <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>

                <h3 style="color: #ef4444; margin-bottom: 1rem; font-family: var(--font-display);">Analysis Failed</h3>

                <p style="color: #fc5555; margin-bottom: 1rem;">${error.message}</p>

                <p style="color: #9ca3af; font-size: 0.9rem;">

                    Please check your CSV file format and try again. Ensure it has  header row and clean data.

                </p>

            </div>

    `;

        resultsSeaction.classList.remove('hidden');

    } finally {

        analyzeBtn.disabled = false;

        analyzeBtn.querySelector('.btn-text').textContent = 'Analyze with AI';

        progressFill.style.width = '0%';

    }

});



// ===================================

// Display Results

// ===================================

/** Normalize smart-dispatch API response to shape expected by displayResults */
function normalizeSmartDispatchResponse(data) {
    if (data.recommended_model_name !== undefined && data.metric_value !== undefined) {
        return data; // already /analyze shape
    }
    const rec = data.recommended_model || {};
    const scenario = data.scenario || {};
    const taskType = data.task_type || 'analysis';
    const topModels = data.top_models || [];
    const firstScore = rec.score ?? topModels[0]?.score;
    const isPercent = typeof firstScore === 'number' && firstScore > 1 && firstScore <= 100;
    return {
        recommended_model: rec.name,
        recommended_model_name: rec.name,
        task_type: taskType,
        metric_value: isPercent ? firstScore / 100 : firstScore,
        score: isPercent ? firstScore / 100 : firstScore,
        resoning: rec.explanation || (topModels[0] && topModels[0].explanation),
        business_insights: scenario.name ? { headline: `Detected scenario: ${scenario.icon || ''} ${scenario.name} (${scenario.industry || ''}). Confidence: ${data.overall_confidence != null ? data.overall_confidence + '%' : 'N/A'}.`, recommended_action: rec.explanation } : undefined,
        model_explanation: typeof rec.explanation === 'object' ? rec.explanation : (rec.explanation ? { how_it_works: rec.explanation } : (topModels[0] && topModels[0].explanation ? { how_it_works: topModels[0].explanation } : undefined)),
        preview_data: [],
        scenario: data.scenario,
        top_models: data.top_models,
        dataset_summary: data.dataset_summary,
        session_id: data.session_id,
        feature_info: data.feature_info,
        scenario_recommended: rec.scenario_recommended || false,
        scenario_name: rec.scenario_name || (scenario && scenario.name) || ''
    };
}

function displayResults(data) {

    const {

        recommended_model,

        recommended_model_name,

        task_type,

        metric_value,

        score,

        resoning,

        business_insights,

        model_explanation,

        preview_data,

        scenario_recommended = false,

        scenario_name = ''

    } = data;



    const modelName = (typeof recommended_model === 'string' ? recommended_model : recommended_model?.name) || recommended_model_name;

    const rawScore = metric_value ?? score;

    const finlScore = typeof rawScore === 'number' && rawScore <= 1 ? rawScore : (typeof rawScore === 'number' ? rawScore / 100 : undefined);



    let html = '';



    // Min Result Crd

    html += `

    <div class="result-crd-min" style = "

background: linear - gradient(135deg, rgb(139, 92, 246, 0.15) 0 %, rgb(59, 130, 246, 0.05) 100 %);

backdrop-filter: blur(20px);

border: 2px solid rgb(139, 92, 246, 0.4);

border-radius: 24px;

padding: 2.5rem;

box - shadow: 0 8px 32px rgb(0, 0, 0, 0.5);

animation: slideIn 0.5s ease - out;

">

    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap;">

                <span style="

                    padding: 0.5rem 1rem;

                    background: var(--gradient-primry);

                    color: white;

                    border-radius: 20px;

                    font-size: 0.85rem;

                    font-weight: 600;

                    text-transform: uppearcse;

                ">${task_type || 'Analysis'}</span>

                <h2 style="

                    font-family: var(--font-display);

                    font-size: 2rem;

                    font-weight: 700;

                    flex: 1;

                    margin: 0;

                ">${modelName}</h2>

                ${scenario_recommended && scenario_name ? `

                <span style="

                    padding: 0.5rem 1rem;

                    background: linear-gradient(135deg, rgb(34, 197, 94, 0.3), rgb(16, 185, 129, 0.2));

                    color: rgb(134, 239, 172);

                    border-radius: 20px;

                    font-size: 0.85rem;

                    font-weight: 600;

                    border: 1px solid rgb(34, 197, 94, 0.5);

                ">✓ Recommended for ${scenario_name}</span>

                ` : ''}

            </div>



    ${resoning ? `

                <p style="color: var(--text-secondary); margin-bottom: 1.5rem; line-height: 1.7;">

                    ${resoning}

                </p>

            ` : ''
        }

            

            ${finlScore !== undefined ? `

                <div style="

                    display: flex;

                    justify-content: space-between;

                    align-items: center;

                    padding: 1.5rem;

                    background: rgb(0, 0, 0, 0.3);

                    border-radius: 16px;

                    margin-bottom: 1.5rem;

                ">

                    <span style="color: var(--text-muted); font-size: 1rem;">Model Performance</span>

                    <span style="

                        font-family: var(--font-display);

                        font-size: 2.5rem;

                        font-weight: 700;

                        background: var(--gradient-success);

                        -webkit-background-clip: text;

                        -webkit-text-fill-color: transparent;

                        background-clip: text;

                    ">${(finlScore * 100).toFixed(2)}%</span>

                </div>

            ` : ''
        }

            

            ${business_insights && business_insights.headline ? `

                <div style="

                    background: rgb(16, 185, 129, 0.1);

                    border-left: 4px solid var(--accent-green);

                    padding: 1.5rem;

                    border-radius: 12px;

                ">

                    <div style="

                        font-weight: 700;

                        color: var(--accent-green);

                        margin-bottom: 0.75rem;

                        display: flex;

                        align-items: center;

                        gap: 0.5rem;

                    ">

                        📊 Business Insights

                    </div>

                    <p style="margin: 0; color: var(--text-secondary); line-height: 1.7;">

                        ${business_insights.headline}

                    </p>

                    ${business_insights.recommended_action ? `

                        <p style="

                            margin-top: 0.75rem;

                            color: var(--accent-green);

                            font-weight: 600;

                        ">

                            💡 ${business_insights.recommended_action}

                        </p>

                    ` : ''}

                </div>

            ` : ''
        }

        </div>

    `;



    // Model Explanation

    if (model_explanation) {

        html += `

    <div class="explanation-crd" style = "

background: linear - gradient(135deg, rgb(236, 72, 153, 0.1) 0 %, rgb(139, 92, 246, 0.1) 100 %);

backdrop-filter: blur(20px);

border: 2px solid rgb(236, 72, 153, 0.3);

border-radius: 24px;

padding: 2.5rem;

box - shadow: 0 8px 32px rgb(0, 0, 0, 0.4);

animation: slideIn 0.5s ease - out 0.2s backwards;

">

    <h3 style="

font - family: var(--font - display);

font - size: 1.75rem;

margin - bottom: 1.5rem;

display: flex;

align - items: center;

gap: 0.75rem;

">

    < span >💡</span>

        Understanding Your Model

                </h3>



    ${model_explanation.task_context ? `

                    <div style="

                        background: rgb(59, 130, 246, 0.1);

                        padding: 1rem;

                        border-radius: 12px;

                        margin-bottom: 1.5rem;

                        border-left: 4px solid var(--accent-blue);

                    ">

                        <p style="margin: 0; font-weight: 600;">${model_explanation.task_context}</p>

                    </div>

                ` : ''
            }

                

                ${model_explanation.analogy ? `

                    <h4 style="margin-bottom: 0.75rem; font-size: 1.25rem;">${model_explanation.analogy}</h4>

                ` : ''
            }

                

                ${model_explanation.how_it_works ? `

                    <p style="color: var(--text-secondary); line-height: 1.7; margin-bottom: 1rem;">

                        ${model_explanation.how_it_works}

                    </p>

                ` : ''
            }

                

                ${model_explanation.real_world_example ? `

                    <div style="

                        background: rgb(0, 0, 0, 0.3);

                        border-left: 4px solid var(--accent-pink);

                        padding: 1.5rem;

                        border-radius: 12px;

                        margin: 1.5rem 0;

                    ">

                        ${model_explanation.real_world_example}

                    </div>

                ` : ''
            }

                

                ${model_explanation.best_for ? `

                    <p style="

                        margin-top: 1.5rem;

                        padding-top: 1.5rem;

                        border-top: 1px solid var(--border-color);

                        color: var(--accent-green);

                        font-weight: 600;

                    ">

                        ✅ Best for: ${model_explanation.best_for}

                    </p>

                ` : ''
            }

            </div>

    `;

    }



    // Preview Dt

    if (preview_data && preview_data.length > 0) {

        html += `

    <div class="data-preview-crd" style = "

background: rgb(17, 24, 39, 0.6);

backdrop-filter: blur(20px);

border: 1px solid var(--border-color);

border-radius: 24px;

padding: 2rem;

overflow: hidden;

animation: slideIn 0.5s ease - out 0.4s backwards;

">

    <h4 style="

font - family: var(--font - display);

margin - bottom: 1.5rem;

display: flex;

align - items: center;

gap: 0.75rem;

font - size: 1.5rem;

">

    < span >📋</span>

        Dt Preview

                </h4>

    <div style="overflow-x: auto;">

        <table style="

                        width: 100%;

                        border-collapse: collapse;

                        font-size: 0.9rem;

                    ">

            <thead>

                <tr style="background: rgb(139, 92, 246, 0.1);">

                    ${Object.keys(preview_data[0]).map(key => `

                                    <th style="

                                        padding: 1rem;

                                        text-align: left;

                                        border-bottom: 2px solid var(--border-color);

                                        font-weight: 600;

                                        white-space: nowrap;

                                    ">${key}</th>

                                `).join('')}

                </tr>

            </thead>

            <tbody>

                ${preview_data.map((row, idx) => `

                                <tr style="background: ${idx % 2 === 0 ? 'rgb(0,0,0,0.2)' : 'transparent'};">

                                    ${Object.values(row).map(val => `

                                        <td style="

                                            padding: 0.875rem 1rem;

                                            border-bottom: 1px solid rgb(139, 92, 246, 0.05);

                                            color: var(--text-secondary);

                                        ">${val}</td>

                                    `).join('')}

                                </tr>

                            `).join('')}

            </tbody>

        </table>

    </div>

            </div>

    `;

    }



    resultsContent.innerHTML = html;

    resultsSeaction.classList.remove('hidden');

}



// Reaset button

resetBtn.addEventListener('click', () => {

    resultsSeaction.classList.add('hidden');

    resetFileUpload();

    uploadForm.reset();



    // Scroll to upload seaction

    document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });

});



// ===================================

// Notifiactions

// ===================================



function showNotification(message, type = 'info') {

    const notifiaction = document.createElement('div');

    notifiaction.className = 'notifiaction';



    const colors = {

        success: 'rgb(16, 185, 129, 0.9)',

        error: 'rgb(239, 68, 68, 0.9)',

        warning: 'rgb(245, 158, 11, 0.9)',

        info: 'rgb(59, 130, 246, 0.9)'

    };



    notifiaction.style.cssText = `

position: fixed;

top: 6rem;

right: 2rem;

background: ${colors[type]};

color: white;

padding: 1rem 1.5rem;

border-radius: 12px;

box - shadow: 0 8px 32px rgb(0, 0, 0, 0.5);

z-index: 10000;

font - weight: 600;

animation: slideInRight 0.3s ease - out;

max - width: 400px;

`;



    notifiaction.textContent = message;

    document.body.appendChild(notifiaction);



    setTimeout(() => {

        notifiaction.style.animation = 'slideOutRight 0.3s ease-out';

        setTimeout(() => notifiaction.remove(), 300);

    }, 4000);

}



// Add animation styles

const style = document.createElement('style');

style.textContent = `

@keyframes slideInRight {

        from {

        opacity: 0;

        transform: translateX(100 %);

    }

        to {

        opacity: 1;

        transform: translateX(0);

    }

}



@keyframes slideOutRight {

        from {

        opacity: 1;

        transform: translateX(0);

    }

        to {

        opacity: 0;

        transform: translateX(100 %);

    }

}

`;

document.head.appendChild(style);



// ===================================

// Initialize

// ===================================



document.addEventListener('DOMContentLoaded', () => {

    console.log('🤖 AutoML Intelligence Platform initialized');



    // Initialize particle system

    const canvas = document.getElementById('particleCnvs');

    if (canvas) {

        new ParticleSystem(canvas);

        console.log('✨ Particle system activated');

    }



    // Initialize typing effect

    const typingElement = document.getElementById('typingText');

    if (typingElement) {

        const texts = [

            'Upload your CSV and let AI find the perfect model aautomatically',

            'Powered by 10+ machine learning algorithms with rel-world explanations',

            'From classifiaction to clustering - we handle it ll intelligently'

        ];

        new TypingEffect(typingElement, texts);

        console.log('⌨️ Typing animation starated');

    }



    console.log('📡 API Bse URL:', API_BASE_URL);



    // Load scenarios showcse

    loadScenarios();



    console.log('✅ All systems ready!');

});



// ===================================

// Scenario Showcase Loader (NEW!)

// ===================================



async function loadScenarios() {

    const scenariosGrid = document.getElementById('scenariosGrid');

    if (!scenariosGrid) return;



    try {

        const response = await fetch(`${API_BASE_URL}/scenarios`);

        const data = await response.json();



        if (data.scenarios && data.scenarios.length > 0) {

            renderScenarios(data.scenarios);

        } else {

            scenariosGrid.innerHTML = '\u003cp\u003eNo scenarios available\u003c/p\u003e';

        }

    } catch (error) {

        console.error('Failed to load scenarios:', error);

        scenariosGrid.innerHTML = '\u003cp\u003eFailed to load scenarios. Please refresh the page.\u003c/p\u003e';

    }

}



function renderScenarios(scenarios) {

    const scenariosGrid = document.getElementById('scenariosGrid');



    scenariosGrid.innerHTML = scenarios.map((scenario, index) => `

        <div class="scenario-card" onclick="loadScenarioSample('${scenario.id}')" data-scenario="${scenario.id}">

            <div class="scenario-header">

                <div class="scenario-icon">${scenario.icon || '🤖'}</div>

                <span class="scenario-badge">${scenario.task || 'Analysis'}</span>

            </div>

            <div class="scenario-content">

                <h3 class="scenario-title">${scenario.name}</h3>

                <p class="scenario-description">${scenario.description}</p>

            </div>

            <div class="scenario-footer">

                <span class="scenario-industry">${scenario.industry || 'General'}</span>

                <span class="scenario-action">Try Sample <span class="scenario-action-icon">→</span></span>

            </div>

        </div>

    `).join('');



    console.log(`✅ Loaded ${scenarios.length} scenarios`);

}



function loadScenarioSample(scenarioId) {

    console.log(`🎯 Loading sample for scenario: ${scenarioId}`);



    // Mp scenario ID to sample file

    const scenarioFileMap = {

        'crypto_signals': 'crypto_signals.csv',

        'loan_applications': 'loan_applications.csv',

        'sms_spam': 'sms_spam.csv',

        'banknote_authentication': 'banknote_authentication.csv',

        'heart_disease': 'heart_disease.csv',

        'customer_churn': 'customer_churn.csv',

        'marketing_roi': 'marketing_roi.csv',

        'used_car_prices': 'used_car_prices.csv',

        'airbnb_pricing': 'airbnb_pricing.csv',

        'flight_delays': 'flight_delays.csv',

        'color_palette': 'color_palette.csv',

        'stock_sectors': 'stock_sectors.csv',

        'credit_card_transactions': 'credit_card_transactions.csv'

    };



    const sampleFile = scenarioFileMap[scenarioId];

    if (!sampleFile) {

        showNotification('⚠️ Sample data not available for this scenario', 'warning');

        return;

    }



    // Fetch and load the sample file

    fetch(`/samples/${sampleFile}`)

        .then(response => {

            if (!response.ok) {

                // Fallback: try from backend samples endpoint

                return fetch(`${API_BASE_URL}/samples/${sampleFile}`);

            }

            return response;

        })

        .then(response => response.blob())

        .then(blob => {

            // Create File object from the blob

            const file = new File([blob], sampleFile, { type: 'text/csv' });



            // Simulate file selection

            const dataTransfer = new DataTransfer();

            dataTransfer.items.add(file);

            fileInput.files = dataTransfer.files;



            // Trigger file select handler

            selectedFile = file;

            displayFileInfo(file);



            // Scroll to upload section

            document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });



            showNotification(`✅ Loaded ${sampleFile}! Click "Analyze with AI" to start`, 'success');

        })

        .catch(error => {

            console.error('Failed to load sample:', error);

            showNotification('❌ Failed to load sample data. Please upload manually.', 'error');

        });

}

// ===================================



// Demo Modal and Routing System (NEW!)



// ===================================







// Globl state for demo system



const demoStte = {



    sessionId: null,



    scenarioId: null,



    scenarioDt: null,



    featureInfo: null,



    taskType: null,



    modelName: null



};







// Show demo modal fter analysis



function showDemoModal(analysisDt) {



    const {



        session_id,



        scenario,



        recommended_model,



        top_models,



        overall_confidence



    } = analysisDt;







    // Store in globl state



    demoStte.sessionId = session_id;



    demoStte.scenarioId = scenario.id;



    demoStte.scenarioDt = scenario;



    demoStte.featureInfo = analysisDt.feature_info;



    demoStte.taskType = analysisDt.task_type;



    demoStte.modelName = recommended_model?.name || 'Unknown';







    // Update modal content



    document.getElementById('modalScenarioIcon').textContent = scenario.icon || '✅x✅ ';



    document.getElementById('modalScenarioName').textContent = scenario.name || 'Scenario Detected';



    document.getElementById('modalScenarioDescription').textContent = scenario.description || '';



    document.getElementById('modalModelName').textContent = recommended_model?.name || 'Unknown';







    // Format score based on task type



    let scoreDisplay = '';



    if (recommended_model) {



        if (analysisDt.task_type === 'classifiaction') {



            scoreDisplay = `${recommended_model.score} % `;



        } else {



            scoreDisplay = recommended_model.score.toFixed(2);



        }



    }



    document.getElementById('modalModelScore').textContent = scoreDisplay;



    document.getElementById('modalConfidence').textContent = `${scenario.confidence} % `;







    // Show modal



    document.getElementById('demoModalOverlay').classList.remove('hidden');



}







// Close demo modal



function closeDemoModal() {



    document.getElementById('demoModalOverlay').classList.add('hidden');



}







// Navigate to demo page



function navigateToDemo() {



    if (!demoStte.sessionId || !demoStte.scenarioId) {



        showNotification('❌ No demo session available', 'warning');



        return;



    }







    // Hide modal



    closeDemoModal();







    // Hide min seactions



    document.querySelector('.hero').style.display = 'none';



    document.getElementById('features').style.display = 'none';



    document.getElementById('scenarios').style.display = 'none';



    document.getElementById('upload').style.display = 'none';



    document.getElementById('results').style.display = 'none';



    document.getElementById('models').style.display = 'none';







    // Show demo container



    const demoContiner = document.getElementById('demoPageContainer');



    demoContiner.classList.remove('hidden');







    // Render the demo page



    renderDemoPage(demoStte.scenarioId);







    // Scroll to top



    window.scrollTo({ top: 0, behavior: 'smooth' });



}







// Go bck to min lb



function backToLab() {



    // Hide demo container



    document.getElementById('demoPageContainer').classList.add('hidden');







    // Show min seactions



    document.querySelector('.hero').style.display = '';



    document.getElementById('features').style.display = '';



    document.getElementById('scenarios').style.display = '';



    document.getElementById('upload').style.display = '';



    document.getElementById('results').style.display = '';



    document.getElementById('models').style.display = '';







    // Scroll to results



    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });



}







// Download results s JSON



function downloadResults() {



    // Get current results data from the page



    const resultsDt = {



        session_id: demoStte.sessionId,



        scenario: demoStte.scenarioDt,



        model: demoStte.modelName,



        task_type: demoStte.taskType,



        timestamp: new Date().toISOString()



    };







    // Crete blob and download



    const blob = new Blob([JSON.stringify(resultsDt, null, 2)], { type: 'application/json' });



    const url = URL.createObjectURL(blob);



    const anchor = document.createElement('');



    anchor.href = url;



    anchor.download = `automl - results - ${Date.now()}.json`;



    document.body.appendChild(anchor);



    anchor.click();



    document.body.removeChild(anchor);



    URL.revokeObjectURL(url);







    showNotification('✅Results downloaded!', 'success');



    closeDemoModal();



}







// Mke prediction with cached model



async function makeLivePrediction(inputFetures) {



    try {



        const response = await fetch(`${API_BASE_URL} / demo / predict / ${demoStte.sessionId}`, {



            method: 'POST',



            headers: {



                'Content-Type': 'application/json'



            },



            body: JSON.stringify({ features: inputFetures })



        });







        if (!response.ok) {



            const error = await response.json();



            throw new Error(error.detail?.message || error.detail || 'Prediction failed');



        }







        const result = await response.json();



        return result;



    } catch (error) {



        console.error('Prediction error:', error);



        throw error;



    }



}







// Render demo page based on scenario



function renderDemoPage(scenarioId) {



    const demoContiner = document.getElementById('demoPageContainer');







    // Common demo page header



    const headerHTML = `



    < div style = "



            background: rgb(17, 24, 39, 0.8);



backdrop - filter: blur(20px);



border - bottom: 1px solid var(--border - color);



padding: 1.5rem 2rem;



position: sticky;



top: 0;



z - index: 1000;



">



    < div style = "max-width: 1400px; margin: 0 auto; display: flex; align-items: center; justify-content: space-between;" >



                <div style="display: flex; align-items: center; gap: 1rem;">



                    <span style="font-size: 2rem;">${demoStte.scenarioDt.icon}</span>



                    <div>



                        <h2 style="margin: 0; font-family: var(--font-display); font-size: 1.5rem;">${demoStte.scenarioDt.name}</h2>



                        <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">${demoStte.modelName} ⬅️ ${demoStte.taskType}</p>



                    </div>



                </div>



                <button onclick="backToLab()" style="



                    padding: 0.75rem 1.5rem;



                    background: rgb(139, 92, 246, 0.1);



                    border: 2px solid var(--border-color);



                    border-radius: 12px;



                    color: var(--text-primary);



                    font-weight: 600;



                    cursor: pointer;



                    transition: var(--transition-base);



                ">



                    ⬅️ Bck to Model Lb



                </button>



            </div >



        </div >



    `;







    // Render scenario-specific demo



    let demoContent = '';







    switch (scenarioId) {



        case 'crypto_signals':



            demoContent = renderCryptoDemo();



            break;



        case 'loan_applications':



            demoContent = renderLoanDemo();



            break;



        case 'sms_spam':



            demoContent = renderSpamDemo();



            break;



        default:



            demoContent = renderGenericDemo(scenarioId);



    }







    demoContiner.innerHTML = headerHTML + demoContent;



}







// Event listeners for modal buttons



document.addEventListener('DOMContentLoaded', () => {



    document.getElementById('btnContinueToDemo').addEventListener('click', navigateToDemo);



    document.getElementById('btnDownloadResults').addEventListener('click', downloadResults);



    document.getElementById('btnCloseDemoModal').addEventListener('click', closeDemoModal);







    // Close modal when clicking overly



    document.getElementById('demoModalOverlay').addEventListener('click', (e) => {



        if (e.target.id === 'demoModalOverlay') {



            closeDemoModal();



        }

    });



});



// ===================================



// Demo Page Renderers (Proof of Concept)



// ===================================







// 1. Crypto Trding Signl Demo



function renderCryptoDemo() {



    return `



    < div style = "max-width: 1400px; margin: 0 auto; padding: 3rem 2rem;" >



        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">



            <!-- Input Panel -->



            <div style="



                    background: rgb(17, 24, 39, 0.8);



                    backdrop-filter: blur(20px);



                    border: 2px solid var(--border-color);



                    border-radius: 24px;



                    padding: 2rem;



                ">



                <h3 style="font-family: var(--font-display); font-size: 1.5rem; margin-bottom: 1.5rem;">



                    ✅ Enter Technical Indicators



                </h3>







                <form id="cryptoForm" onsubmit="handleCryptoPrediction(event)" style="display: flex; flex-direction: column; gap: 1rem;">



                    <div>



                        <label style="display: block; margin-bottom: 0.5rem; color: var(--text-muted); font-size: 0.9rem;">RSI (0-100)</label>



                        <input type="number" name="rsi" step="0.01" required



                            style="width: 100%; padding: 0.75rem; background: rgb(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); font-size: 1rem;"



                            placeholder="e.g., 65.5">



                    </div>







                    <div>



                        <label style="display: block; margin-bottom: 0.5rem; color: var(--text-muted); font-size: 0.9rem;">MACD</label>



                        <input type="number" name="macd" step="0.001" required



                            style="width: 100%; padding: 0.75rem; background: rgb(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); font-size: 1rem;"



                            placeholder="e.g., 0.05">



                    </div>







                    <div>



                        <label style="display: block; margin-bottom: 0.5rem; color: var(--text-muted); font-size: 0.9rem;">Volume</label>



                        <input type="number" name="volume" required



                            style="width: 100%; padding: 0.75rem; background: rgb(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); font-size: 1rem;"



                            placeholder="e.g., 1000000">



                    </div>







                    <button type="submit" style="



                            margin-top: 1rem;



                            padding: 1rem;



                            background: var(--gradient-primry);



                            border: none;



                            border-radius: 12px;



                            color: white;



                            font-weight: 700;



                            font-size: 1.05rem;



                            cursor: pointer;



                            transition: var(--transition-base);



                        ">



                        ❌  Get Live Prediction



                    </button>



                </form>



            </div>







            <!-- Results Panel -->



            <div id="cryptoResults" style="



                    background: rgb(17, 24, 39, 0.8);



                    backdrop-filter: blur(20px);



                    border: 2px solid var(--border-color);



                    border-radius: 24px;



                    padding: 2rem;



                    display: flex;



                    align-items: center;



                    justify-content: center;



                    min-height: 400px;



                ">



                <p style="color: var(--text-muted); text-align: center;">



                    Enter technical indicators and get  live buy/sell signal



                </p>



            </div>



        </div>



        </div >



    `;



}







// 2. Loan Approval Demo



function renderLoanDemo() {



    return `



    < div style = "max-width: 1000px; margin: 0 auto; padding: 3rem 2rem;" >



        <div style="



                background: rgb(17, 24, 39, 0.8);



                backdrop-filter: blur(20px);



                border: 2px solid var(--border-color);



                border-radius: 24px;



                padding: 3rem;



            ">



            <h3 style="font-family: var(--font-display); font-size: 1.75rem; margin-bottom: 2rem; text-align: center;">



                ✅x✅✅ Loan Application Assessment



            </h3>







            <form id="loanForm" onsubmit="handleLoanPrediction(event)" style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">



                <div>



                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-muted); font-size: 0.9rem;">Annual Income ($)</label>



                    <input type="number" name="income" required



                        style="width: 100%; padding: 0.75rem; background: rgb(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); font-size: 1rem;"



                        placeholder="e.g., 75000">



                </div>







                <div>



                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-muted); font-size: 0.9rem;">Credit Score</label>



                    <input type="number" name="credit_score" min="300" max="850" required



                        style="width: 100%; padding: 0.75rem; background: rgb(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); font-size: 1rem;"



                        placeholder="e.g., 720">



                </div>







                <div>



                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-muted); font-size: 0.9rem;">Loan Amount ($)</label>



                    <input type="number" name="loan_amount" required



                        style="width: 100%; padding: 0.75rem; background: rgb(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); font-size: 1rem;"



                        placeholder="e.g., 250000">



                </div>







                <div>



                    <label style="display: block; margin-bottom: 0.5rem; color: var(--text-muted); font-size: 0.9rem;">Employment Length (years)</label>



                    <input type="number" name="employment_length" required



                        style="width: 100%; padding: 0.75rem; background: rgb(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 8px; color: var(--text-primary); font-size: 1rem;"



                        placeholder="e.g., 5">



                </div>







                <div style="grid-column: 1 / -1;">



                    <button type="submit" style="



                            width: 100%;



                            padding: 1rem;



                            background: var(--gradient-primry);



                            border: none;



                            border-radius: 12px;



                            color: white;



                            font-weight: 700;



                            font-size: 1.05rem;



                            cursor: pointer;



                            transition: var(--transition-base);



                        ">



                        ✅x ✅ Check Loan Eligibility



                    </button>



                </div>



            </form>







            <div id="loanResults" style="margin-top: 2rem;"></div>



        </div>



        </div >



    `;



}







// 3. SMS Spam Detector Demo



function renderSpamDemo() {



    return `



    < div style = "max-width: 900px; margin: 0 auto; padding: 3rem 2rem;" >



        <div style="



                background: rgb(17, 24, 39, 0.8);



                backdrop-filter: blur(20px);



                border: 2px solid var(--border-color);



                border-radius: 24px;



                padding: 3rem;



            ">



            <h3 style="font-family: var(--font-display); font-size: 1.75rem; margin-bottom: 1.5rem; text-align: center;">



                ✅x ✅ SMS Spam Deteaction



            </h3>







            <form id="spamForm" onsubmit="handleSpamPrediction(event)">



                <div style="margin-bottom: 1.5rem;">



                    <label style="display: block; margin-bottom: 0.75rem; color: var(--text-muted); font-size: 0.9rem;">



                        Enter  message to analyze



                    </label>



                    <textarea name="message" rows="6" required



                        style="width: 100%; padding: 1rem; background: rgb(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 12px; color: var(--text-primary); font-size: 1rem; resize: vertical; font-family: var(--font-primry);"



                        placeholder="e.g., Free entry in 2  weekly comp to win FA Cup finl tickets 21st My 2005. Text FA to 87121 to receive entry question(std txt rate)"></textarea>



                </div>







                <button type="submit" style="



                        width: 100%;



                        padding: 1rem;



                        background: var(--gradient-primry);



                        border: none;



                        border-radius: 12px;



                        color: white;



                        font-weight: 700;



                        font-size: 1.05rem;



                        cursor: pointer;



                        transition: var(--transition-base);



                    ">



                    ✅x ✅ Analyze Message



                </button>



            </form>







            <div id="spamResults" style="margin-top: 2rem;"></div>



        </div>



        </div >



    `;



}







// Generic demo for other scenarios



function renderGenericDemo(scenarioId) {



    return `



    < div style = "max-width: 1000px; margin: 0 auto; padding: 3rem 2rem; text-align: center;" >



        <div style="



                background: rgb(17, 24, 39, 0.8);



                backdrop-filter: blur(20px);



                border: 2px solid var(--border-color);



                border-radius: 24px;



                padding: 4rem;



            ">



            <div style="font-size: 4rem; margin-bottom: 1rem;">${demoStte.scenarioDt.icon}</div>



            <h3 style="font-family: var(--font-display); font-size: 2rem; margin-bottom: 1rem;">



                ${demoStte.scenarioDt.name}



            </h3>



            <p style="color: var(--text-muted); font-size: 1.1rem; margin-bottom: 2rem;">



                Interactive demo for this scenario is coming soon!



            </p>



            <p style="color: var(--text-secondary); max-width: 600px; margin: 0 auto;">



                This scenario has been successfully analyzed. The model is ready, but the interactive demo



                interface is currently under development.



            </p>



        </div>



        </div >



    `;



}







// ===================================



// Demo Prediction Handlers



// ===================================







async function handleCryptoPrediction(event) {



    event.preventDefault();







    const formData = new FormData(event.target);



    const features = {



        rsi: parseFloat(formData.get('rsi')),



        macd: parseFloat(formData.get('macd')),



        volume: parseFloat(formData.get('volume'))



    };







    // Show loading



    document.getElementById('cryptoResults').innerHTML = `



    < div style = "text-align: center;" >



            <div class="loading-spinner" style="margin: 0 auto 1rem;"></div>



            <p style="color: var(--text-muted);">Analyzing market conditions...</p>



        </div >



    `;







    try {



        const result = await makeLivePrediction(features);







        // Determine signal color



        const isBuy = result.prediction === 1 || result.prediction === 'BUY';



        const signalColor = isBuy ? 'var(--accent-green)' : 'var(--accent-pink)';



        const signalText = isBuy ? 'BUY' : 'SELL';



        const confidencePercent = (result.confidence * 100).toFixed(1);







        document.getElementById('cryptoResults').innerHTML = `



    < div style = "text-align: center;" >



                <div style="



                    font-size: 4rem;



                    font-weight: 700;



                    color: ${signalColor};



                    margin-bottom: 1rem;



                    text-shadow: 0 0 30px ${signalColor}80;



                ">



                    ${signalText}



                </div>



                <div style="margin-bottom: 2rem;">



                    <div style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem;">Confidence</div>



                    <div style="font-family: var(--font-display); font-size: 2.5rem; font-weight: 700; color: ${signalColor};">



                        ${confidencePercent}%



                    </div>



                </div>



                <div style="



                    padding: 1.5rem;



                    background: rgb(${isBuy ? '16, 185, 129' : '236, 72, 153'}, 0.1);



                    border-left: 4px solid ${signalColor};



                    border-radius: 12px;



                    text-align: left;



                ">



                    <p style="margin: 0; color: var(--text-secondary);">



                        ${result.explanation || `Model recommends ${signalText} signal with ${confidencePercent}% confidence`}



                    </p>



                </div>

</div > `;



    } catch (error) {



        document.getElementById('cryptoResults').innerHTML = `



    < div style = "text-align: center; color: var(--accent-pink);" >



                <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>



                <p>${error.message}</p>



            </div >



    `;



    }



}







async function handleLoanPrediction(event) {



    event.preventDefault();







    const formData = new FormData(event.target);



    const features = {



        income: parseFloat(formData.get('income')),



        credit_score: parseFloat(formData.get('credit_score')),



        loan_amount: parseFloat(formData.get('loan_amount')),



        employment_length: parseFloat(formData.get('employment_length'))



    };







    // Show loading



    document.getElementById('loanResults').innerHTML = `



    < div style = "text-align: center; padding: 2rem;" >



            <div class="loading-spinner" style="margin: 0 auto 1rem;"></div>



            <p style="color: var(--text-muted);">Assessing application...</p>



        </div >



    `;







    try {



        const result = await makeLivePrediction(features);







        const isApproved = result.prediction === 1 || result.prediction === 'APPROVED';



        const statusColor = isApproved ? 'var(--accent-green)' : 'var(--accent-pink)';



        const statusText = isApproved ? 'APPROVED' : 'DENIED';



        const confidencePercent = (result.confidence * 100).toFixed(1);







        document.getElementById('loanResults').innerHTML = `



    < div style = "
padding: 2rem;
background: rgb(${isApproved ? '16, 185, 129' : '239, 68, 68'}, 0.1);
border: 2px solid ${statusColor};
border - radius: 16px;
text - align: center;
">
    < div style = "font-size: 3.5rem; margin-bottom: 1rem;" > ${isApproved ? '✅' : '❌'}</div >



                <div style="font-family: var(--font-display); font-size: 2rem; font-weight: 700; color: ${statusColor}; margin-bottom: 1rem;">



                    ${statusText}



                </div>



                <div style="color: var(--text-secondary); margin-bottom: 1rem;">



                    Confidence: ${confidencePercent}%



                </div>



                <p style="color: var(--text-muted);">



                    ${result.explanation || `Application ${statusText.toLowerCase()} with ${confidencePercent}% confidence`}



                </p>



            </div >



    `;



    } catch (error) {



        document.getElementById('loanResults').innerHTML = `



    < div style = "text-align: center; padding: 2rem; color: var(--accent-pink);" >



                <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>



                <p>${error.message}</p>



            </div >



    `;



    }



}







async function handleSpamPrediction(event) {



    event.preventDefault();







    const formData = new FormData(event.target);



    const features = {



        message: formData.get('message')



    };







    // Show loading



    document.getElementById('spamResults').innerHTML = `



    < div style = "text-align: center; padding: 2rem;" >
        <div>
            <div class="loading-spinner" style="margin: 0 auto 1rem;"></div>
            <p style="color: var(--text-muted);">Analyzing message...</p>
        </div>
    </div >



    `;







    try {



        const result = await makeLivePrediction(features);







        const isSpam = result.prediction === 1 || result.prediction === 'SPAM';



        const statusColor = isSpam ? 'var(--accent-pink)' : 'var(--accent-green)';



        const statusText = isSpam ? 'SPAM' : 'HAM (Legitimate)';



        const confidencePercent = (result.confidence * 100).toFixed(1);







        document.getElementById('spamResults').innerHTML = `



    < div style = "
padding: 2rem;
background: rgb(${isSpam ? '239, 68, 68' : '16, 185, 129'}, 0.1);
border: 2px solid ${statusColor};
border - radius: 16px;
text - align: center;
">
    < div style = "font-size: 3.5rem; margin-bottom: 1rem;" > ${isSpam ? '❌' : '✅'}</div >



                <div style="font-family: var(--font-display); font-size: 2rem; font-weight: 700; color: ${statusColor}; margin-bottom: 1rem;">



                    ${statusText}



                </div>



                <div style="color: var(--text-secondary); margin-bottom: 1rem;">



                    Confidence: ${confidencePercent}%



                </div>



                <p style="color: var(--text-muted);">



                    ${result.explanation || `Message classified as ${statusText} with ${confidencePercent}% confidence`}



                </p>



            </div >



    `;



    } catch (error) {



        document.getElementById('spamResults').innerHTML = `



    < div style = "text-align: center; padding: 2rem; color: var(--accent-pink);" >



                <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>



                <p>${error.message}</p>



            </div >



    `;



    }



}





