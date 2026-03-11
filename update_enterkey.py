#!/usr/bin/env python3
# Add Enter key functionality and conversation memory to HTML

with open('templates/consultation-notes.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Add Enter key handlers before the closing </script> tag
enter_key_script = '''
        // ENTER KEY HANDLERS - Ctrl+Enter to submit
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
        });
'''

# Find the existing QA section and add conversation memory
old_qa_section = '''        // QA TAB
        document.getElementById('askBtn').addEventListener('click', async () => {
            const question = document.getElementById('clinicalQuestion').value.trim();
            if (!question) {
                document.getElementById('clinicalAnswer').textContent = 'Please enter a question...';
                return;
            }

            const answer = document.getElementById('clinicalAnswer');
            answer.innerHTML = '<span class="loading"></span><span class="loading"></span><span class="loading"></span>';

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                const data = await response.json();
                answer.textContent = data.answer || 'No answer generated';
                
                // Clear input after successful answer
                document.getElementById('clinicalQuestion').value = '';
                document.getElementById('clinicalQuestion').style.height = 'auto';
            } catch (err) {
                answer.textContent = `Error: ${err.message}`;
            }
        });'''

new_qa_section = '''        // QA TAB - WITH CONVERSATION MEMORY FOR FOLLOW-UPS
        let qaHistory = JSON.parse(localStorage.getItem('vividmedi_qa_history')) || [];

        document.getElementById('askBtn').addEventListener('click', async () => {
            const question = document.getElementById('clinicalQuestion').value.trim();
            if (!question) {
                document.getElementById('clinicalAnswer').textContent = 'Please enter a question...';
                return;
            }

            const answer = document.getElementById('clinicalAnswer');
            answer.innerHTML = '<span class="loading"></span><span class="loading"></span><span class="loading"></span>';

            try {
                // Build context from last 3 Q&A for follow-ups
                const context = qaHistory.slice(-3).map((q, i) => `Previous Q: ${q.question}\\nPrevious A: ${q.answer.substring(0, 150)}...`).join('\\n\\n');
                
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question, context: context })
                });
                const data = await response.json();
                answer.textContent = data.answer || 'No answer generated';
                
                // Save to history for future follow-ups
                qaHistory.push({ question: question, answer: data.answer });
                if (qaHistory.length > 10) qaHistory = qaHistory.slice(-10);
                localStorage.setItem('vividmedi_qa_history', JSON.stringify(qaHistory));
                
                // Clear input after successful answer
                document.getElementById('clinicalQuestion').value = '';
                document.getElementById('clinicalQuestion').style.height = 'auto';
            } catch (err) {
                answer.textContent = `Error: ${err.message}`;
            }
        });'''

html = html.replace(old_qa_section, new_qa_section)

# Update clear button to mention history reset
old_clear = '''        document.getElementById('clearQaBtn').addEventListener('click', () => {
            const answer = document.getElementById('clinicalAnswer').textContent;
            if (answer && answer !== 'Answer will appear here...' && answer.trim() && !answer.includes('loading')) {
                window.saveOutput(answer, 'question');
            }
            document.getElementById('clinicalQuestion').value = '';
            document.getElementById('clinicalAnswer').textContent = 'Answer will appear here...';
            document.getElementById('clinicalQuestion').style.height = 'auto';
            document.getElementById('clinicalQuestion').focus();
        });'''

new_clear = '''        document.getElementById('clearQaBtn').addEventListener('click', () => {
            const answer = document.getElementById('clinicalAnswer').textContent;
            if (answer && answer !== 'Answer will appear here...' && answer.trim() && !answer.includes('loading')) {
                window.saveOutput(answer, 'question');
            }
            document.getElementById('clinicalQuestion').value = '';
            document.getElementById('clinicalAnswer').textContent = 'Answer will appear here...';
            document.getElementById('clinicalQuestion').style.height = 'auto';
            // Clear history for fresh start
            qaHistory = [];
            localStorage.removeItem('vividmedi_qa_history');
            document.getElementById('clinicalQuestion').focus();
        });'''

html = html.replace(old_clear, new_clear)

# Insert Enter key handlers before Initialize
html = html.replace('        // Initialize\n        renderSavedOutputs();', 
                   enter_key_script + '\n        // Initialize\n        renderSavedOutputs();')

with open('templates/consultation-notes.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Added Enter key and conversation memory")
