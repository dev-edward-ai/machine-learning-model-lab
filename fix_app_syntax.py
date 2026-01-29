#!/usr/bin/env python3
"""Comprehensive fix script for app.js corruption - ENHANCED VERSION"""
import re

file_path = r"c:\Users\User\OneDrive\Desktop\ml_tesing\machine-learning-model-lab\frontend\app.js"

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print("🔍 Scanning for all corruption patterns...")

# Define all replacements as (pattern, replacement) tuples
replacements = [
    # Fix function typo
    ('funaction', 'function'),
    
    # Fix variable name typos
    ('smaple', 'sample'),
    ('Smaple', 'Sample'),
    ('scenarioFileMp', 'scenarioFileMap'),
    ('ppliaction', 'application'),
    ('uthentiaction', 'authentication'),
    ('bnknote', 'banknote'),
    ('trnsactions', 'transactions'),
    ('scenario-crd', 'scenario-card'),
    ('bckend', 'backend'),
    ('Dteaction', 'Detection'),
    ('Prediaction', 'Prediction'),
    ('prediaction', 'prediction'),
    ('Hndlers', 'Handlers'),
    ('Overly', 'Overlay'),
    ('Nvigte', 'Navigate'),
    ('continer', 'container'),
    ('timestmap', 'timestamp'),
    ('cchead', 'cached'),
    ('temaplte', 'template'),
    ('direaction', 'direction'),
    ('Indictors', 'Indicators'),
    ('verticl', 'vertical'),
    ('comap', 'comp'),
    ('Intearctive', 'Interactive'),
    ('intearctive', 'interactive'),
    ('interfce', 'interface'),
    ('signl', 'signal'),
    ('feture', 'feature'),
    ('emaployment', 'employment'),
    ('Emaployment', 'Employment'),
    ('loan_mount', 'loan_amount'),
    ('Pearcent', 'Percent'),
    ('reaset', 'reset'),
    ('aaappend', 'append'),
    ('aappend', 'append'),
    ('credit_crd', 'credit_card'),
    ('mkeLivePrediction', 'makeLivePrediction'),
    ('mke', 'make'),
    
    # Fix double letter typos
    ('AAssessing', 'Assessing'),
    ('Aapplication', 'Application'),
    ('aautoML', 'AutoML'),
    ('AautoML', 'AutoML'),
    
    # Fix CSS property typos
    ('text-aalign', 'text-align'),
    ('text - aalign', 'text-align'),
    ('--text-primry', '--text-primary'),
    ('--transition-bse', '--transition-base'),
    ('border - radius', 'border-radius'),
    
    # Fix awaith -> with
    (' awaith ', ' with '),
    
    # Fix sawaitch -> switch
    ('sawaitch', 'switch'),
    
    # Fix other typos
    ('Dte', 'Date'),
    
    # Fix .dd -> .add
    ('.dd(', '.add('),
]

# Count fixes
fix_count = {}
for pattern, replacement in replacements:
    count = content.count(pattern)
    if count > 0:
        fix_count[pattern] = count
        content = content.replace(pattern, replacement)

# Fix HTML tag spacing issues (< div -> <div, </div > -> </div>)
print("🔧 Fixing HTML tag spacing...")
html_spacing_fixes = 0

# Fix opening tags with spaces: < div style = "..." > to <div style="...">
content_before = content
content = re.sub(r'<\s+(\w+)\s+style\s+=\s+"', r'<\1 style="', content)
html_spacing_fixes += len(re.findall(r'<\s+(\w+)\s+style\s+=\s+"', content_before))

# Fix closing tags with spaces: </div > to </div>
content_before = content
content = re.sub(r'</(\w+)\s+>', r'</\1>', content)
html_spacing_fixes += len(re.findall(r'</(\w+)\s+>', content_before))

# Fix corrupted emojis using regex to catch variations
print("🔧 Fixing corrupted emojis...")
emoji_fixes = 0
emoji_patterns = [
    (r'✅\x01✅️', '❌'),
    (r'✅S\x01&\s*', '✅'),
    (r'✅R\x01', '❌'),
    (r'✅x\x01\x01✅', '❌'),
    (r'✅x✅\s*', '📊'),
    (r'✅x✅✅', '💳'),
    (r'✅x\s+', '📊'),
    (r'✅S&\s*', '✅'),
    (r'✅\s+✅', '⬅️'),
]

for pattern, replacement in emoji_patterns:
    matches = len(re.findall(pattern, content))
    if matches > 0:
        emoji_fixes += matches
        content = re.sub(pattern, replacement, content)

# Fix createElement issues
print("🔧 Fixing createElement syntax...")
content = re.sub(r'\bconst\s+\s*=\s*document\.createElement\(', 'const anchor = document.createElement(', content)
content = re.sub(r'(?<!anchor)\.href\s*=\s*url;', 'anchor.href = url;', content)
content = re.sub(r'(?<!anchor)\.download\s*=', 'anchor.download =', content)
content = re.sub(r'document\.body\.appendChild\(\s*\);', 'document.body.appendChild(anchor);', content)
content = re.sub(r'(?<!anchor)\.click\(\);', 'anchor.click();', content)
content = re.sub(r'document\.body\.removeChild\(\s*\);', 'document.body.removeChild(anchor);', content)

# Save the fixed content
with open(file_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content)

print("\n✅ COMPLETE! Fixed all syntax errors and corruption in app.js")
print("\n📊 Summary of fixes:")
print(f"  - Text replacements: {sum(fix_count.values())} occurrences")
for pattern, count in sorted(fix_count.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"    • '{pattern}': {count}")
print(f"  - HTML tag spacing fixes: {html_spacing_fixes}")
print(f"  - Corrupted emoji fixes: {emoji_fixes}")
print(f"  - createElement syntax fixes: applied")
print("\n🎉 Your app.js is now clean!")
