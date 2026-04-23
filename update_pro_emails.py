#!/usr/bin/env python3
import re

with open('app.py', 'r') as f:
    content = f.read()

# Add PRO_EMAILS support after CREATOR_EMAIL line
old = '''CREATOR_EMAIL = (os.getenv("CREATOR_EMAIL") or "").strip().lower()

if STRIPE_SECRET_KEY:'''

new = '''CREATOR_EMAIL = (os.getenv("CREATOR_EMAIL") or "").strip().lower()
# Additional PRO emails (comma-separated)
PRO_EMAILS_STR = (os.getenv("PRO_EMAILS") or "").lower()
PRO_EMAILS_LIST = {e.strip() for e in PRO_EMAILS_STR.split(",") if e.strip()}
if CREATOR_EMAIL:
    PRO_EMAILS_LIST.add(CREATOR_EMAIL)

if STRIPE_SECRET_KEY:'''

content = content.replace(old, new)

# Update the auth check to use PRO_EMAILS_LIST
old_auth = '''        if CREATOR_EMAIL and email == CREATOR_EMAIL and user.get("plan") != "pro":
            upgrade_user_to_pro(user["id"])
            user["plan"] = "pro"'''

new_auth = '''        if email in PRO_EMAILS_LIST and user.get("plan") != "pro":
            upgrade_user_to_pro(user["id"])
            user["plan"] = "pro"'''

content = content.replace(old_auth, new_auth)

with open('app.py', 'w') as f:
    f.write(content)

print("✓ Updated app.py to support multiple PRO emails")
