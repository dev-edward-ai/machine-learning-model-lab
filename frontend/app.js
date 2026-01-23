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

            // Mouse interaction
            const dx = this.mouseX - particle.x;
            const dy = this.mouseY - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < 100) {
                const force = (100 - distance) / 100;
                particle.vx -= (dx / distance) * force * 0.1;
                particle.vy -= (dy / distance) * force * 0.1;
            }

            // Wrap around screen
            if (particle.x < 0) particle.x = this.canvas.width;
            if (particle.x > this.canvas.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.canvas.height;
            if (particle.y > this.canvas.height) particle.y = 0;

            // Draw particle
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(139, 92, 246, ${particle.opacity})`;
            this.ctx.fill();

            // Draw connections
            this.particles.slice(i + 1).forEach(particle2 => {
                const dx = particle.x - particle2.x;
                const dy = particle.y - particle2.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 100) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(particle.x, particle.y);
                    this.ctx.lineTo(particle2.x, particle2.y);
                    this.ctx.strokeStyle = `rgba(139, 92, 246, ${0.15 * (1 - distance / 100)})`;
                    this.ctx.lineWidth = 1;
                    this.ctx.stroke();
                }
            });
        });

        requestAnimationFrame(() => this.animate());
    }
}

// ===================================
// Typing Animation
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

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
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
    if (e.target !== fileRemove && !fileRemove.contains(e.target)) {
        fileInput.click();
    }
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
        fileInput.files = files;
        handleFileSelect({ target: { files } });
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
const resultsSection = document.getElementById('results');
const resultsContent = document.getElementById('resultsContent');
const resetBtn = document.getElementById('resetBtn');

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!selectedFile) {
        showNotification('⚠️ Please select a CSV file first', 'warning');
        return;
    }

    // Show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing...';
    statusIndicator.classList.remove('hidden');
    statusText.textContent = 'Processing your dataset...';
    uploadProgress.classList.remove('hidden');
    progressFill.style.width = '20%';

    try {
        // Prepare form data
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
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });

        progressFill.style.width = '70%';

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail?.error || errorData.detail || 'Analysis failed');
        }

        const data = await response.json();

        // Complete progress
        progressFill.style.width = '100%';
        statusText.textContent = 'Analysis complete!';

        // Display results
        setTimeout(() => {
            displayResults(data);
            uploadProgress.classList.add('hidden');
            statusIndicator.classList.add('hidden');

            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });

            showNotification('✅ Analysis completed successfully!', 'success');
        }, 500);

    } catch (error) {
        console.error('Error:', error);
        uploadProgress.classList.add('hidden');
        statusIndicator.classList.add('hidden');

        showNotification(`❌ ${error.message}`, 'error');

        resultsContent.innerHTML = `
            <div class="error-card" style="
                background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
                border: 2px solid rgba(239, 68, 68, 0.3);
                border-radius: 20px;
                padding: 2rem;
                text-align: center;
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
                <h3 style="color: #ef4444; margin-bottom: 1rem; font-family: var(--font-display);">Analysis Failed</h3>
                <p style="color: #fca5a5; margin-bottom: 1rem;">${error.message}</p>
                <p style="color: #9ca3af; font-size: 0.9rem;">
                    Please check your CSV file format and try again. Ensure it has a header row and clean data.
                </p>
            </div>
        `;
        resultsSection.classList.remove('hidden');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('.btn-text').textContent = 'Analyze with AI';
        progressFill.style.width = '0%';
    }
});

// ===================================
// Display Results
// ===================================

function displayResults(data) {
    const {
        recommended_model,
        recommended_model_name,
        task_type,
        metric_value,
        score,
        reasoning,
        business_insights,
        model_explanation,
        preview_data
    } = data;

    const modelName = recommended_model || recommended_model_name;
    const finalScore = metric_value || score;

    let html = '';

    // Main Result Card
    html += `
        <div class="result-card-main" style="
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%);
            backdrop-filter: blur(20px);
            border: 2px solid rgba(139, 92, 246, 0.4);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            animation: slideIn 0.5s ease-out;
        ">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
                <span style="
                    padding: 0.5rem 1rem;
                    background: var(--gradient-primary);
                    color: white;
                    border-radius: 20px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    text-transform: uppercase;
                ">${task_type || 'Analysis'}</span>
                <h2 style="
                    font-family: var(--font-display);
                    font-size: 2rem;
                    font-weight: 700;
                    flex: 1;
                    margin: 0;
                ">${modelName}</h2>
            </div>
            
            ${reasoning ? `
                <p style="color: var(--text-secondary); margin-bottom: 1.5rem; line-height: 1.7;">
                    ${reasoning}
                </p>
            ` : ''}
            
            ${finalScore !== undefined ? `
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1.5rem;
                    background: rgba(0, 0, 0, 0.3);
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
                    ">${(finalScore * 100).toFixed(2)}%</span>
                </div>
            ` : ''}
            
            ${business_insights && business_insights.headline ? `
                <div style="
                    background: rgba(16, 185, 129, 0.1);
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
            ` : ''}
        </div>
    `;

    // Model Explanation
    if (model_explanation) {
        html += `
            <div class="explanation-card" style="
                background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                backdrop-filter: blur(20px);
                border: 2px solid rgba(236, 72, 153, 0.3);
                border-radius: 24px;
                padding: 2.5rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
                animation: slideIn 0.5s ease-out 0.2s backwards;
            ">
                <h3 style="
                    font-family: var(--font-display);
                    font-size: 1.75rem;
                    margin-bottom: 1.5rem;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                ">
                    <span>💡</span>
                    Understanding Your Model
                </h3>
                
                ${model_explanation.task_context ? `
                    <div style="
                        background: rgba(59, 130, 246, 0.1);
                        padding: 1rem;
                        border-radius: 12px;
                        margin-bottom: 1.5rem;
                        border-left: 4px solid var(--accent-blue);
                    ">
                        <p style="margin: 0; font-weight: 600;">${model_explanation.task_context}</p>
                    </div>
                ` : ''}
                
                ${model_explanation.analogy ? `
                    <h4 style="margin-bottom: 0.75rem; font-size: 1.25rem;">${model_explanation.analogy}</h4>
                ` : ''}
                
                ${model_explanation.how_it_works ? `
                    <p style="color: var(--text-secondary); line-height: 1.7; margin-bottom: 1rem;">
                        ${model_explanation.how_it_works}
                    </p>
                ` : ''}
                
                ${model_explanation.real_world_example ? `
                    <div style="
                        background: rgba(0, 0, 0, 0.3);
                        border-left: 4px solid var(--accent-pink);
                        padding: 1.5rem;
                        border-radius: 12px;
                        margin: 1.5rem 0;
                    ">
                        ${model_explanation.real_world_example}
                    </div>
                ` : ''}
                
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
                ` : ''}
            </div>
        `;
    }

    // Preview Data
    if (preview_data && preview_data.length > 0) {
        html += `
            <div class="data-preview-card" style="
                background: rgba(17, 24, 39, 0.6);
                backdrop-filter: blur(20px);
                border: 1px solid var(--border-color);
                border-radius: 24px;
                padding: 2rem;
                overflow: hidden;
                animation: slideIn 0.5s ease-out 0.4s backwards;
            ">
                <h4 style="
                    font-family: var(--font-display);
                    margin-bottom: 1.5rem;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    font-size: 1.5rem;
                ">
                    <span>📋</span>
                    Data Preview
                </h4>
                <div style="overflow-x: auto;">
                    <table style="
                        width: 100%;
                        border-collapse: collapse;
                        font-size: 0.9rem;
                    ">
                        <thead>
                            <tr style="background: rgba(139, 92, 246, 0.1);">
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
                                <tr style="background: ${idx % 2 === 0 ? 'rgba(0,0,0,0.2)' : 'transparent'};">
                                    ${Object.values(row).map(val => `
                                        <td style="
                                            padding: 0.875rem 1rem;
                                            border-bottom: 1px solid rgba(139, 92, 246, 0.05);
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
    resultsSection.classList.remove('hidden');
}

// Reset button
resetBtn.addEventListener('click', () => {
    resultsSection.classList.add('hidden');
    resetFileUpload();
    uploadForm.reset();

    // Scroll to upload section
    document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
});

// ===================================
// Notifications
// ===================================

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = 'notification';

    const colors = {
        success: 'rgba(16, 185, 129, 0.9)',
        error: 'rgba(239, 68, 68, 0.9)',
        warning: 'rgba(245, 158, 11, 0.9)',
        info: 'rgba(59, 130, 246, 0.9)'
    };

    notification.style.cssText = `
        position: fixed;
        top: 6rem;
        right: 2rem;
        background: ${colors[type]};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        z-index: 10000;
        font-weight: 600;
        animation: slideInRight 0.3s ease-out;
        max-width: 400px;
    `;

    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// Add animation styles
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100%);
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
            transform: translateX(100%);
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
    const canvas = document.getElementById('particleCanvas');
    if (canvas) {
        new ParticleSystem(canvas);
        console.log('✨ Particle system activated');
    }

    // Initialize typing effect
    const typingElement = document.getElementById('typingText');
    if (typingElement) {
        const texts = [
            'Upload your CSV and let AI find the perfect model automatically',
            'Powered by 10+ machine learning algorithms with real-world explanations',
            'From classification to clustering - we handle it all intelligently'
        ];
        new TypingEffect(typingElement, texts);
        console.log('⌨️ Typing animation started');
    }

    console.log('📡 API Base URL:', API_BASE_URL);
    console.log('✅ All systems ready!');
});
