#!/usr/bin/env python3
with open('templates/consultation-notes.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Replace the Ctrl+Enter handlers with plain Enter
old_handlers = '''        // ENTER KEY HANDLERS - Ctrl+Enter to submit
        document.getElementById('clinicalInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                document.getElementById('convertBtn').click();
            }
        });

        document.getElementById('clinicalQuestion').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                document.getElementById('askBtn').click();
            }
        });'''

new_handlers = '''        // ENTER KEY HANDLERS - Enter to submit, Shift+Enter for new line
        document.getElementById('clinicalInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.getElementById('convertBtn').click();
            }
        });

        document.getElementById('clinicalQuestion').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.getElementById('askBtn').click();
            }
        });'''

html = html.replace(old_handlers, new_handlers)

with open('templates/consultation-notes.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Updated: Enter to submit, Shift+Enter for new line")
