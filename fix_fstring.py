with open('app.py', 'r') as f:
    lines = f.readlines()

# Fix line 892 - the broken f-string
for i, line in enumerate(lines):
    if 'user_content = f"Generate from this clinical data:' in line:
        lines[i] = '        user_content = f"Generate from this clinical data:\\n\\n{clinical_data}"\n'
        break

with open('app.py', 'w') as f:
    f.writelines(lines)

print("Fixed f-string")
