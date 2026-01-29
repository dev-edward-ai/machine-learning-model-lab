#!/usr/bin/env python3
"""Fix remaining red line errors - spacing and typos"""
import re

file_path = r"c:\Users\User\OneDrive\Desktop\ml_tesing\machine-learning-model-lab\frontend\app.js"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print("🔍 Scanning for remaining issues...")

fixes_applied = {}

# Fix 1: HTML closing tag spacing - space before >
before = content.count('" >')
content = content.replace('" >', '">')
fixes_applied['HTML closing tag spacing (\" >)'] = before - content.count('" >')

# Fix 2: Typo - Legitimte -> Legitimate
before = content.count('Legitimte')
content = content.replace('Legitimte', 'Legitimate')
fixes_applied['Legitimte → Legitimate'] = before - content.count('Legitimte')

# Fix 3: Typo - "classified s" -> "classified as"
before = content.count('classified s ')
content = content.replace('classified s ', 'classified as ')
fixes_applied['classified s → classified as'] = before - content.count('classified s ')

# Fix 4: Extra space in closing div tags "> ${"
before = content.count('> ${')
content = content.replace('> ${', '>${')
fixes_applied['Closing tag spacing (> ${)'] = before - content.count('> ${')

# Fix 5: Extra closing </div> tag in spam section
# Looking for pattern where we have duplicate closing tags
pattern = r'</div>\s*</div>\s*\n\s*`;\s*\n\s*\n\s*\n\s*try'
if re.search(pattern, content):
    # This looks like a duplicate, let's fix it
    content = re.sub(
        r'(<div class="loading-spinner".*?</div>\s*<p.*?</p>\s*</div>\s*)(</div>)',
        r'\1',
        content,
        flags=re.DOTALL
    )
    fixes_applied['Removed duplicate closing tags'] = 1

with open(file_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content)

print("\n✅ Fixed remaining red line errors:")
total = 0
for issue, count in fixes_applied.items():
    if count > 0:
        print(f"  • {issue}: {count}")
        total += count

print(f"\n📊 Total fixes: {total}")
print("🎉 All red lines should now be fixed!")
