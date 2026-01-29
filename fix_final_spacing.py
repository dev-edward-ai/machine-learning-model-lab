#!/usr/bin/env python3
"""Final fix for remaining class attribute spacing"""
import re

file_path = r"c:\Users\User\OneDrive\Desktop\ml_tesing\machine-learning-model-lab\frontend\app.js"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Count before
before_count = len(re.findall(r'<\s+div\s+class\s+=\s+"', content))

# Fix class attribute spacing: < div class = " -> <div class="
content = re.sub(r'<\s+div\s+class\s+=\s+"', '<div class="', content)

# Count after
after_count = len(re.findall(r'<\s+div\s+class\s+=\s+"', content))

with open(file_path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content)

print(f"✅ Fixed {before_count - after_count} additional HTML class spacing issues")
print(f"Remaining: {after_count}")
