#!/usr/bin/env python3
"""
Fix corrupted app.js file by removing null bytes and fixing encoding issues.
"""

import os
import re

# Read the corrupted file
input_file = r'frontend\app.js'
backup_file = r'frontend\app.js.backup'
output_file = r'frontend\app.js.fixed'

print(f"Reading corrupted file: {input_file}")

# Create backup
with open(input_file, 'rb') as f:
    corrupted_content = f.read()

# Save backup
with open(backup_file, 'wb') as f:
    f.write(corrupted_content)
print(f"✅ Backup created: {backup_file}")

# Remove all null bytes
print("Removing null bytes...")
cleaned_content = corrupted_content.replace(b'\x00', b'')

# Try to decode with UTF-8, replace errors
try:
    text_content = cleaned_content.decode('utf-8', errors='replace')
    print(f"✅ Decoded to text ({len(text_content)} chars)")
except Exception as e:
    print(f"❌ Error decoding: {e}")
    exit(1)

# Fix corrupted emojis - replace � (replacement character) with appropriate emojis
# Based on context clues from the code
fixes = [
    ('�a����', '⚠️'),  # Warning emoji for error states
    ('�S& ', '🎉'),    # Success emoji
    ('�', '✅'),        # Generic replacement for corrupted checkmarks/success
]

for old, new in fixes:
    if old in text_content:
        text_content = text_content.replace(old, new)
        print(f"✅ Fixed emoji: {repr(old)} → {new}")

# Write fixed content
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(text_content)

print(f"\n✅ Fixed file written: {output_file}")
print(f"📊 Original size: {len(corrupted_content)} bytes")
print(f"📊 Cleaned size: {len(text_content.encode('utf-8'))} bytes")
print(f"📊 Removed ~{len(corrupted_content) - len(text_content.encode('utf-8'))} bytes of corruption")

# Count lines
lines = text_content.split('\n')
print(f"📊 Total lines: {len(lines)}")

print("\n✨ File repair complete!")
print(f"\nNext steps:")
print(f"1. Review the fixed file:{output_file}")
print(f"2. If it looks good, replace the original:")
print(f"   mv {output_file} {input_file}")
