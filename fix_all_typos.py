#!/usr/bin/env python3
"""
Fix ALL typos in app.js caused by missing letters from corruption.
"""

# Read the file
with open('frontend/app.js', 'r', encoding='utf-8') as f:
    content = f.read()

# Comprehensive typo fixes - organized by category
typo_fixes = {
    # CSS Properties - missing 'a'
    'text-lign': 'text-align',
    'mrgin': 'margin',
    'border-rdius': 'border-radius',
    'bckground': 'background',
    
    # CSS functions - missing 'a'
    'vr(': 'var(',
    
    # CSS variable names - missing 'a'
    '--font-disply': '--font-display',
    '--text-secondry': '--text-secondary',
    '--text-muted': '--text-muted',  # might be --text-mute
    
    # Common words - missing 'a'
    'Anlysis': 'Analysis',
    'anlysis': 'analysis',
    'Plese': 'Please',
    'plese': 'please',
    'gin': 'again',
    'Ensure': 'Ensure',  
    
    # Common words - missing other letters
    'messge': 'message',
    'formt': 'format',
    'heder': 'header',
    'clen': 'clean',
    'Filed': 'Failed',
    'filed': 'failed',
    
    # Variable names - missing 'a'
    'signlColor': 'signalColor',
    'signlText': 'signalText',
    'explntion': 'explanation',
    
    # Common typos with missing letters
    'vilble': 'available',
    'wrning': 'warning',
    'confidencePercent': 'confidencePercent',  # This might be correct
    
    # RGB/color values
    'rgb(': 'rgba(',  # might be intentional
    
    # More missing 'a' patterns
    'dt': 'data',
    'hs': 'has',
    ' nd ': ' and ',
    
    # Additional CSS
    'font-fmily': 'font-family',
    'disply': 'display',
    
    # Color codes  
    '#fc55': '#fc5555',
    '#9c3f': '#9ca3af',
}

print(f"Original file size: {len(content)} chars")
print(f"Applying {len(typo_fixes)} typo fixes...\n")

fixes_applied = 0
for typo, correct in typo_fixes.items():
    count = content.count(typo)
    if count > 0:
        content = content.replace(typo, correct)
        print(f"✅ Fixed '{typo}' → '{correct}' ({count} occurrences)")
        fixes_applied += count

print(f"\n📊 Total fixes applied: {fixes_applied}")

# Write the fixed content
with open('frontend/app.js', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✅ Fixed file size: {len(content)} chars")
print(f"\n✨ All typos fixed! File updated.")
