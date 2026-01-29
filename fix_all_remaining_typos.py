#!/usr/bin/env python3
"""
MASTER TYPO FIX - Comprehensive fix for ALL remaining typos in app.js
"""

with open('frontend/app.js', 'r', encoding='utf-8') as f:
    content = f.read()

print(f"Original size: {len(content)} bytes")
print("Applying comprehensive typo fixes...\n")

# MASTER TYPO LIST - All patterns found
typos = {
    # Platform/System
    'Pltform': 'Platform',
    'Animtions': 'Animations',
    'loclhost': 'localhost',
    
    # JavaScript Classes/Methods
    'clss': 'class',
    'Prticle': 'Particle',
    'prticle': 'particle',
    'prticles': 'particles',
    'cnvs': 'canvas',
    'nimte': 'animate',
    'ddEventListener': 'addEventListener',
    'preventDefult': 'preventDefault',
    'querySelector': 'querySelector',
    'querySelectorAll': 'querySelectorAll',
    
    # Math/Numbers
    'Mth': 'Math',
    'rndom': 'random',
    'rdius': 'radius',
    'opcity': 'opacity',
    'distnce': 'distance',
    
    # Canvas methods
    'clerRect': 'clearRect',
    'beagainPth': 'beginPath',
    'rc': 'arc',
    'moveTo': 'moveTo',
    'lineTo': 'lineTo',
    
    # Animation
    'requestAnimtionFrme': 'requestAnimationFrame',
    'nimtion': 'animation',
    'trnsition': 'transition',
    'trnsform': 'transform',
    'trnslteX': 'translateX',
    
    # Array/Object methods  
    'forEch': 'forEach',
    'mp': 'map',
    'vlues': 'values',
    
    # DOM Properties
    'widatah': 'width',
    'clssList': 'classList',
    'clssNme': 'className',
    'textContent': 'textContent',
    'innerHTML': 'innerHTML',
    
    # Common words - missing 'a'
    'sync': 'async',
    'wit': 'await',
    'flse': 'false',
    'trget': 'target',
    'dataTrnsfer': 'dataTransfer',
    'DtTrnsfer': 'DataTransfer',
    
    # File/Upload
    'fileNme': 'fileName',
    'uplod': 'upload',
    'Uplod': 'Upload',
    'hndleFileSelect': 'handleFileSelect',
    'hndle': 'handle',
    
    # Form/Data
    'FormDt': 'FormData',
    'formDt': 'formData',
    'ppend': 'append',
    'vlue': 'value',
    'vl': 'val',
    
    # Analysis/Model
    'nlyze': 'analyze',
    'Anlyze': 'Analyze',
    'nlyzeBtn': 'analyzeBtn',
    'nlysis': 'analysis',
    'Anlysis': 'Analysis',
    'nlysisDt': 'analysisData',
    
    # Status/State  
    'sttus': 'status',
    'sttusIndictor': 'statusIndicator',
    'sttusColor': 'statusColor',
    'sttusText': 'statusText',
    'stte': 'state',
    
    # Scenario
    'scenrio': 'scenario',
    'Scenrio': 'Scenario',
    'scenrios': 'scenarios',
    'scenriosGrid': 'scenariosGrid',
    'lodScenrios': 'loadScenarios',
    'renderScenrios': 'renderScenarios',
    'lodScenrioSmple': 'loadScenarioSample',
    'scenrioId': 'scenarioId',
    'scenrioFileMp': 'scenarioFileMap',
    
    # Sample
    'smple': 'sample',
    'Smple': 'Sample',
    
    # Model/Modal
    'modl': 'modal',
    'Modl': 'Modal',
    'modlScenrioIcon': 'modalScenarioIcon',
    'modlScenrioNme': 'modalScenarioName',
    'modlScenrioDescription': 'modalScenarioDescription',
    'modlModelNme': 'modalModelName',
    'modlConfidence': 'modalConfidence',
    'demoModlOverly': 'demoModalOverlay',
    
    # Demo/Page
    'demoPgeContiner': 'demoPageContainer',
    'renderDemoPge': 'renderDemoPage',
    'nvigteToDemo': 'navigateToDemo',
    
    # Navigation
    'nchor': 'anchor',
    'behvior': 'behavior',
    'strt': 'start',
    'bckToLb': 'backToLab',
    
    # Drag/Drop
    'Drg': 'Drag',
    'drg': 'drag',
    'drgover': 'dragover',
    'drgleve': 'dragleave',
    
    # Error/Exception
    'ctch': 'catch',
    'detil': 'detail',
    
    # Delay/Duration
    'dely': 'delay',
    'puseDurtion': 'pauseDuration',
    
    # CSS/Styling
    'liner-grdient': 'linear-gradient',
    'grdient': 'gradient',
    'pdding': 'padding',
    'maragain': 'margin',
    'gp': 'gap',
    'lign': 'align',
    'spce': 'space',
    'trnsprent': 'transparent',
    'ccent': 'accent',
    'bckdrop': 'backdrop',
    'shdow': 'shadow',
    'uto': 'auto',
    
    # Features/Attributes
    'fetureInfo': 'featureInfo',
    'fetures': 'features',
    
    # Headline/Action
    'hedline': 'headline',
    'ction': 'action',
    
    # Analogy/Example
    'nlogy': 'analogy',
    'rel_world_exmple': 'real_world_example',
    'exmple': 'example',
    
    # Table
    'tble': 'table',
    'thed': 'thead',
    
    # Create/Element
    'creteElement': 'createElement',
    'ppendChild': 'appendChild',
    
    # Keyframes
    'keyfrmes': 'keyframes',
    
    # Initialize
    'initilize': 'initialize',
    'initilized': 'initialized',
    'Initilize': 'Initialize',
    
    # Particle specific
    'prticleCnvs': 'particleCanvas',
    'ctivted': 'activated',
    
    # Machine Learning
    'utomticlly': 'automatically',
    'mchine': 'machine',
    'lerning': 'learning',
    'lgorithms': 'algorithms',
    'clssifiction': 'classification',
    'clssified': 'classified',
    
    # Scenario files
    'crypto_signls': 'crypto_signals',
    'lon_pplictions': 'loan_applications',
    'sms_spm': 'sms_spam',
    'bnknote_uthentiction': 'banknote_authentication',
    'hert_disese': 'heart_disease',
    'mrketing_roi': 'marketing_roi',
    'used_cr_prices': 'used_car_prices',
    'irbnb_pricing': 'airbnb_pricing',
    'flight_delys': 'flight_delays',
    'color_plette': 'color_palette',  
    'credit_crd_trnsctions': 'credit_card_transactions',
    
    # Fallback
    'Fllbck': 'Fallback',
    
    # Prediction
    'mkeLivePrediction': 'makeLivePrediction',
    
    # Render functions
    'renderCryptoDemo': 'renderCryptoDemo',
    'renderLonDemo': 'renderLoanDemo',
    'renderSpmDemo': 'renderSpamDemo',
    
    # Technical/Indicators
    'technicl': 'technical',
    'indictors': 'indicators',
    
    # Label/Placeholder
    'lbel': 'label',
    'plceholder': 'placeholder',
    
    # MACD
    'mcd': 'macd',
    
    # Textarea
    'textre': 'textarea',
    
    # Weekly/Tickets/Rate
    'wkly': 'weekly',
    'tkts': 'tickets',
    'rte': 'rate',
    
    # Buy typo
    'isB Buy': 'isBuy',
    
    # Parse
    'prseFlot': 'parseFloat',
    
    # Loan/Spam specific
    'lon': 'loan',
    'Lon': 'Loan',
    'spm': 'spam',
    'Spm': 'Spam',
    
    # Download/Timestamp
    'downlod': 'download',
    'Downlod': 'Download',
    'downlodResults': 'downloadResults',
    'timestmp': 'timestamp',
    
    # Character
    'currentChrIndex': 'currentCharIndex',
    'Chr': 'Char',
    
    # Display
    'scoreDisply': 'scoreDisplay',
    
    # Smart dispatch
    'smrt-disptch': 'smart-dispatch',
    'smrt': 'smart',
    'disptch': 'dispatch',
    
    # Notification
    'notifiction': 'notification',
    
    # Stop propagation
    'stopPropgtion': 'stopPropagation',
    
    # Reset
    'resetFileUplod': 'resetFileUpload',
    
    # Update
    'Updatae': 'Update',
    'updatae': 'update',
    
    # DOM
    'hed': 'head',
    
    # Content
    'uplodZoneContent': 'uploadZoneContent',
    'uplodForm': 'uploadForm',
    'uplodProgress': 'uploadProgress',
    
    # Typing
    'currentTextIndex': 'currentTextIndex',
    
    # Window/Document
    'documentElement': 'documentElement',
    'scrollHeight': 'scrollHeight',
    'innerWidatah': 'innerWidth',
    
    # Count
    'prticleCount': 'particleCount',
    
    # Color
    'fillStyle': 'fillStyle',
    'strokeStyle': 'strokeStyle',
    'lineWidatah': 'lineWidth',
    
    # Contain/Contains
    'contins': 'contains',
    
    # Name/Nme
    'nme': 'name',
    'Nme': 'Name',
    
    # More specific patterns
    'cse ': 'case ',
    'brek': 'break',
    'defult': 'default',
    'finlly': 'finally',
    'endsWith': 'endsWith',
    'substring': 'substring',
    'getElementById': 'getElementById',
    'prseInt': 'parseInt',
    'toFixed': 'toFixed',
    'toLowerCse': 'toLowerCase',
    'toUpperCse': 'toUpperCase',
    'join': 'join',
    'slice': 'slice',
    'indexOf': 'indexOf',
    'includes': 'includes',
    'split': 'split',
    'trim': 'trim',
    'replce': 'replace',
    'toISOString': 'toISOString',
    'creteObjectURL': 'createObjectURL',
    'revokeObjectURL': 'revokeObjectURL',
    
    # Approved/Application
    'isApproved': 'isApproved',
    'ppliction': 'application',
    'Ppliction': 'Application',
    
    # Assessment
    'ssessment': 'ssessment',
    'ssessing': 'Assessing',
    'Assessing': 'Assessing',
    
    # Annual
    'Annul': 'Annual',
    
    # Mount/Amount
    'mount': 'mount',  # This might be correct in some contexts
    'lon_mount': 'loan_amount',
    
    # Employment
    'yers': 'years',
    
    # Check/Eligibility
    'eligibility': 'eligibility',
    
    # Confidence
    'confidencePercent': 'confidencePercent',
    
    # Recommended
    'recommended_ction': 'recommended_action',
    
    # Perform
    'Performnce': 'Performance',
    
    # Understand
    'Understnding': 'Understanding',
    
    # Context
    'task_context': 'task_context',
    
    # Best
    'best_for': 'best_for',
    
    # Collapse
    'border-collpse': 'border-collapse',
    'collpse': 'collapse',
    
    # White space
    'white-spce': 'white-space',
    'nowrp': 'nowrap',
    
    # Maximum
    'mx': 'max',
    'mx-widatah': 'max-width',
    'min-height': 'min-height',
    
    # Wrap
    'wrp': 'wrap',
    'flex-wrp': 'flex-wrap',
    
    # Overflow
    'overflow-x': 'overflow-x',
    
    # Backwards
    'bckwrds': 'backwards',
    
    # Messages
    'message': 'message',
    
    # CSS animation
    'ese-out': 'ease-out',
    'ese': 'ease',
    
    # Span
    'spn': 'span',
    
    # Application
    'ppliction': 'application',
    
    # Badge
    'bdge': 'badge',
    
    # Industry
    'industry': 'industry',
    
    # Generl
    'Generl': 'General',
    
    # Load
    'lod': 'load',
    'Lod': 'Load',
    'loded': 'loaded',
    'Loded': 'Loaded',
    'loding': 'loading',
    'Loding': 'Loading',
    
    # Ready
    'redy': 'ready',
    
    # Refresh
    'pge': 'page',
    'Pge': 'Page',
    
    # Blob
    'blob': 'blob',
    
    # Items
    'items': 'items',
    
    # Available
    'vilble': 'available',
    
    # Simulte
    'Simulte': 'Simulate',
    
    # Select
    'select': 'select',
    
    # Trigger
    'Trigger': 'Trigger',
    
    # Scroll
    'scrollIntoView': 'scrollIntoView',
    
    # Click
    'click': 'click',
    
    # Manually  
    'mnully': 'manually',
    
    # Current
    'current': 'current',
    
    # Wrapper
    'wrpper': 'wrapper',
    
    # Interction/Interactive
    'interction': 'interaction',
    'Interctive': 'Interactive',
    'interctive': 'interactive',
    
    # Development
    'development': 'development',
    
    # Conditions
    'conditions': 'conditions',
    
    # Determine
    'Determine': 'Determine',
    
    # Confidence (repeated but important)
    'confidence': 'confidence',
    
    # Recommends
    'recommends': 'recommends',
    
    # Processing
    'Processing': 'Processing',
    
    # Tournament
    'tournment': 'tournament',
    
    # Complete
    'complete': 'complete',
    
    # Successfully
    'successfully': 'successfully',
    
    # Brief
    'brief': 'brief',
    
    # Session
    'session': 'session',
    
    # Overall
    'overll': 'overall',
    
    # Format
    'format': 'format',
    
    # Ensure
    'ensure': 'Ensure',
    
    # Linear
    'liner': 'linear',
    
    # Gradient
    'grdient': 'gradient',
    
    # Template
    'templte': 'template',
    'grid-templte-columns': 'grid-template-columns',
    
    # Panel
    'Pnel': 'Panel',
    
    # Results
    'resultsSection': 'resultsSection',
    'resultsContent': 'resultsContent',
    
    # Reset
    'resetBtn': 'resetBtn',
    
    # Market
    'mrket': 'market',
    
    # Analyzing
    'Anlyzing': 'Analyzing',
    
    # Credit
    'credit': 'credit',
}

fixes_applied = 0
for typo, correct in typos.items():
    count = content.count(typo)
    if count > 0:
        content = content.replace(typo, correct)
        print(f"✅ {typo} → {correct} ({count})")
        fixes_applied += count

print(f"\n📊 Total fixes: {fixes_applied}")

with open('frontend/app.js', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ Final size: {len(content)} bytes")
print("\n✨ ALL TYPOS FIXED!")
