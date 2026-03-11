#!/usr/bin/env python3
with open('templates/consultation-notes.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Replace the auth button section with upgrade button + auth button
old_header_right = '''                <div class="header-right">
                    <div class="theme-toggle">
                        <button class="theme-toggle-btn active" id="darkBtn" data-theme="dark">Dark</button>
                        <button class="theme-toggle-btn" id="lightBtn" data-theme="light">Light</button>
                    </div>
                    <div id="signInDiv"></div>
                    <button class="auth-btn" id="authBtn" style="display:none;">Sign out</button>
                </div>'''

new_header_right = '''                <div class="header-right">
                    <div class="theme-toggle">
                        <button class="theme-toggle-btn active" id="darkBtn" data-theme="dark">Dark</button>
                        <button class="theme-toggle-btn" id="lightBtn" data-theme="light">Light</button>
                    </div>
                    <button class="auth-btn" id="upgradeBtn" style="display:none; background: linear-gradient(135deg, #d4af7a, #c9a561); border: none; box-shadow: 0 4px 15px rgba(201, 165, 97, 0.3);">Upgrade to Pro</button>
                    <div id="signInDiv"></div>
                    <button class="auth-btn" id="authBtn" style="display:none;">Sign out</button>
                </div>'''

html = html.replace(old_header_right, new_header_right)

# Add upgrade button handler before the closing script tag
upgrade_handler = '''
        // Check user plan and show upgrade button
        async function checkUserPlan() {
            try {
                const response = await fetch('/api/me');
                const data = await response.json();
                
                if (data.logged_in && data.plan === 'free') {
                    document.getElementById('upgradeBtn').style.display = 'block';
                } else if (!data.logged_in) {
                    document.getElementById('upgradeBtn').style.display = 'none';
                }
            } catch (err) {
                console.error('Error checking plan:', err);
            }
        }

        // Upgrade button handler
        document.getElementById('upgradeBtn').addEventListener('click', async () => {
            try {
                const response = await fetch('/api/stripe/create-checkout-session', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                if (data.url) {
                    window.location.href = data.url;
                }
            } catch (err) {
                alert('Error creating checkout session: ' + err.message);
            }
        });

        // Check plan on page load
        checkUserPlan();
'''

# Insert before the final closing script tag
html = html.replace('        // Initialize\n        renderSavedOutputs();',
                   upgrade_handler + '\n        // Initialize\n        renderSavedOutputs();')

# Update auth response to check plan
old_auth_response = '''            .then(data => {
                if (data.ok) {
                    localStorage.setItem('auth_token', data.token);
                    localStorage.setItem('user_email', data.user.email);
                    document.getElementById('signInDiv').style.display = 'none';
                    document.getElementById('authBtn').style.display = 'block';
                    alert('Signed in as ' + data.user.email);
                }
            })'''

new_auth_response = '''            .then(data => {
                if (data.ok) {
                    localStorage.setItem('auth_token', data.token);
                    localStorage.setItem('user_email', data.user.email);
                    localStorage.setItem('user_plan', data.user.plan);
                    document.getElementById('signInDiv').style.display = 'none';
                    document.getElementById('authBtn').style.display = 'block';
                    
                    // Show upgrade button if free plan
                    if (data.user.plan === 'free') {
                        document.getElementById('upgradeBtn').style.display = 'block';
                    }
                    
                    alert('Signed in as ' + data.user.email);
                }
            })'''

html = html.replace(old_auth_response, new_auth_response)

with open('templates/consultation-notes.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Added Upgrade to Pro button with Stripe integration")
