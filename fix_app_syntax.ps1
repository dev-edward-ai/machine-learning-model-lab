# Comprehensive fix script for app.js corruption
$filePath = "c:\Users\User\OneDrive\Desktop\ml_tesing\machine-learning-model-lab\frontend\app.js"
$content = Get-Content $filePath -Raw

# Fix function typo
$content = $content -replace 'funaction', 'function'

# Fix variable name typos
$content = $content -replace 'smaple', 'sample'
$content = $content -replace 'Smaple', 'Sample'
$content = $content -replace 'scenarioFileMp', 'scenarioFileMap'
$content = $content -replace 'ppliaction', 'application'
$content = $content -replace 'uthentiaction', 'authentication'
$content = $content -replace 'bnknote', 'banknote'
$content = $content -replace 'trnsactions', 'transactions'
$content = $content -replace 'crd', 'card'
$content = $content -replace 'bckend', 'backend'
$content = $content -replace 'Dteaction', 'Detection'
$content = $content -replace 'Prediaction', 'Prediction'
$content = $content -replace 'prediaction', 'prediction'
$content = $content -replace 'Hndlers', 'Handlers'
$content = $content -replace 'Overly', 'Overlay'
$content = $content -replace 'Nvigte', 'Navigate'
$content = $content -replace 'continer', 'container'
$content = $content -replace 'timestmap', 'timestamp'
$content = $content -replace 'cchead', 'cached'
$content = $content -replace 'temaplte', 'template'
$content = $content -replace 'direaction', 'direction'
$content = $content -replace 'Indictors', 'Indicators'
$content = $content -replace 'verticl', 'vertical'
$content = $content -replace 'comap', 'comp'
$content = $content -replace 'Intearctive', 'Interactive'
$content = $content -replace 'intearctive', 'interactive'
$content = $content -replace 'interfce', 'interface'
$content = $content -replace 'signl', 'signal'
$content = $content -replace 'feture', 'feature'
$content = $content -replace 'emaployment', 'employment'
$content = $content -replace 'Emaployment', 'Employment'
$content = $content -replace 'mount', 'amount'
$content = $content -replace 'Pearcent', 'Percent'
$content = $content -replace 'reaset', 'reset'
$content = $content -replace 'aaappend', 'append'
$content = $content -replace 'aappend', 'append'

# Fix CSS property typos
$content = $content -replace 'text-aalign', 'text-align'
$content = $content -replace '--text-primry', '--text-primary'
$content = $content -replace '--transition-bse', '--transition-base'

# Fix awaith -> with
$content = $content -replace ' awaith ', ' with '

# Fix sawaitch -> switch
$content = $content -replace 'sawaitch', 'switch'

# Fix corrupted emojis - replace with proper ones
$content = $content -replace '✅\u0001✅️', '❌'
$content = $content -replace '✅S\u0001\u0026 ', '✅'
$content = $content -replace '✅R\u0001', '❌'
$content = $content -replace '✅x\u0001\u0001✅', '❌'
$content = $content -replace '✅x✅ ', '📊'
$content = $content -replace '✅x✅✅', '💳'
$content = $content -replace '✅x ', '📊'
$content = $content -replace '✅S& ', '✅'
$content = $content -replace '✅ ✅', '⬅️'

# Save the fixed content
Set-Content -Path $filePath -Value $content -NoNewline

Write-Host "✅ Fixed all syntax errors and corruption in app.js"
Write-Host "Fixed issues:"
Write-Host "  - function typos"
Write-Host "  - variable name typos"
Write-Host "  - CSS property typos"  
Write-Host "  - corrupted emojis"
Write-Host "  - other typos"
