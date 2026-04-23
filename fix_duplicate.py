# Fix the f-string issue - there's a duplicate line
with open('app.py', 'r') as f:
    lines = f.readlines()

# Find and remove the problematic duplicate
new_lines = []
skip_next = False
for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
    
    if 'user_content = f"Generate from this clinical data:' in line and '{clinical_data}"' in line:
        # This is the corrected version, keep it
        new_lines.append('        user_content = f"Generate from this clinical data:\\n\\n{clinical_data}"\n')
        # Check if next line is the duplicate
        if i+1 < len(lines) and '{clinical_data}"' in lines[i+1]:
            skip_next = True
    else:
        new_lines.append(line)

with open('app.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed duplicate line")
