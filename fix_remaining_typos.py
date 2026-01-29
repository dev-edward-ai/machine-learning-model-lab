#!/usr/bin/env python3
"""
Fix remaining typos found in verification.
"""

with open('frontend/app.js', 'r', encoding='utf-8') as f:
    content = f.read()

# Additional typos found
remaining_typos = {
    'Disply': 'Display',
    'Explntion': 'Explanation',
    'Formt': 'Format',
    'bsed': 'based',
    'tsk': 'task',
    'modlModelScore': 'modalModelScore',
    'Anlyze': 'Analyze',
    'Messge': 'Message',
    'clssified': 'classified',
    'sttusText': 'statusText',
    'scoreDisply': 'scoreDisplay',
}

print(f"Applying {len(remaining_typos)} additional fixes...\n")

fixes_applied = 0
for typo, correct in remaining_typos.items():
    count = content.count(typo)
    if count > 0:
        content = content.replace(typo, correct)
        print(f"✅ Fixed '{typo}' → '{correct}' ({count} occurrences)")
        fixes_applied += count

print(f"\n📊 Total additional fixes: {fixes_applied}")

with open('frontend/app.js', 'w', encoding='utf-8') as f:
    f.write(content)

print("✨ All remaining typos fixed!")
