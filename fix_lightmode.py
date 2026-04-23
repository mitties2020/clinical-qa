#!/usr/bin/env python3
with open('templates/consultation-notes.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Fix light mode colors - darker text and better contrast
old_light_theme = '''        /* Light theme */
        html[data-theme="light"] {
            --primary-dark: #f8f9fa;
            --secondary-dark: #ffffff;
            --tertiary-dark: #f0f2f5;
            --text-primary: #1a1a1a;
            --text-secondary: #666666;
            --border-color: #e0e0e0;
        }'''

new_light_theme = '''        /* Light theme */
        html[data-theme="light"] {
            --primary-dark: #fafbfc;
            --secondary-dark: #ffffff;
            --tertiary-dark: #f0f2f5;
            --text-primary: #0a0e27;
            --text-secondary: #3a3a3a;
            --border-color: #d0d5e0;
        }'''

html = html.replace(old_light_theme, new_light_theme)

# Update light mode textarea text color
old_light_textarea = '''        html[data-theme="light"] .input-wrapper:focus-within,
        html[data-theme="light"] .output-wrapper:focus-within {
            background: rgba(240, 242, 245, 1);
        }'''

new_light_textarea = '''        html[data-theme="light"] .input-wrapper:focus-within,
        html[data-theme="light"] .output-wrapper:focus-within {
            background: rgba(240, 242, 245, 1);
        }

        html[data-theme="light"] textarea,
        html[data-theme="light"] .output-text {
            color: #0a0e27;
        }

        html[data-theme="light"] textarea::placeholder {
            color: #666666;
            opacity: 0.8;
        }'''

html = html.replace(old_light_textarea, new_light_textarea)

# Update logo with modern styling
old_logo_style = '''        .logo {
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--metallic-gold), var(--accent-teal));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.5px;
            cursor: pointer;
        }'''

new_logo_style = '''        .logo {
            font-size: 28px;
            font-weight: 800;
            background: linear-gradient(135deg, #d4af7a 0%, #c9a561 50%, #e6d5a8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -1px;
            cursor: pointer;
            text-shadow: 0 2px 8px rgba(201, 165, 97, 0.2);
            filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1));
        }

        html[data-theme="light"] .logo {
            background: linear-gradient(135deg, #a67c4f 0%, #8b6f47 50%, #c9a561 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }'''

html = html.replace(old_logo_style, new_logo_style)

with open('templates/consultation-notes.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Fixed light mode text contrast")
print("Updated logo to modern gold styling")
