#!/usr/bin/env python3
with open('templates/consultation-notes.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Fix the upgrade button text color - change to dark text
old_upgrade_btn = '''                    <button class="auth-btn" id="upgradeBtn" style="display:none; background: linear-gradient(135deg, #d4af7a, #c9a561); border: none; box-shadow: 0 4px 15px rgba(201, 165, 97, 0.3);">Upgrade to Pro</button>'''

new_upgrade_btn = '''                    <button class="auth-btn" id="upgradeBtn" style="display:none; background: linear-gradient(135deg, #d4af7a, #c9a561); border: none; box-shadow: 0 4px 15px rgba(201, 165, 97, 0.3); color: #0a0e27; font-weight: 600;">Upgrade to Pro</button>'''

html = html.replace(old_upgrade_btn, new_upgrade_btn)

with open('templates/consultation-notes.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Fixed upgrade button text color")
