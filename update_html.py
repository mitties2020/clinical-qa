#!/usr/bin/env python3
# -*- coding: utf-8 -*-

with open('templates/consultation-notes.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Find and replace the input options section with dropdown
old_section = '''                    <div class="section">
                        <div class="section-title">Clinical Data Input</div>
                        <div class="input-options">
                            <button class="option-btn active" data-type="text">Text</button>
                            <button class="option-btn" data-type="voice">Voice Upload</button>
                            <button class="option-btn" data-type="file">File Upload</button>
                        </div>
                    </div>'''

new_section = '''                    <div class="section">
                        <div class="section-title">Clinical Data Input</div>
                        <div style="display: flex; gap: 16px; margin-bottom: 16px;">
                            <div style="flex: 1;">
                                <label style="display: block; font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; font-weight: 500;">Report Type</label>
                                <select id="noteTypeSelect" style="width: 100%; padding: 10px; background: rgba(37, 45, 61, 0.6); border: 1px solid rgba(201, 165, 97, 0.3); border-radius: 6px; color: var(--text-primary); font-size: 14px; cursor: pointer;">
                                    <option value="consultation_note">Consultation Note</option>
                                    <option value="referral_letter">Referral Letter</option>
                                    <option value="discharge_summary">Discharge Summary</option>
                                    <option value="progress_note">Progress Note</option>
                                    <option value="clinical_report">Clinical Report</option>
                                    <option value="soap_note">SOAP Note</option>
                                    <option value="procedure_note">Procedure Note</option>
                                </select>
                            </div>
                            <div style="flex: 1;">
                                <label style="display: block; font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; font-weight: 500;">Input Format</label>
                                <div class="input-options" style="margin: 0;">
                                    <button class="option-btn active" data-type="text" style="flex: 1;">Text</button>
                                    <button class="option-btn" data-type="voice" style="flex: 1;">Voice</button>
                                    <button class="option-btn" data-type="file" style="flex: 1;">File</button>
                                </div>
                            </div>
                        </div>
                    </div>'''

if old_section in html:
    html = html.replace(old_section, new_section)
    print("Replaced section")

# Update the convert button handler to include note type
old_fetch = "body: JSON.stringify({ clinical_data: input })"
new_fetch = "body: JSON.stringify({ clinical_data: input, note_type: document.getElementById('noteTypeSelect').value })"

if old_fetch in html:
    html = html.replace(old_fetch, new_fetch)
    print("Updated fetch call")

with open('templates/consultation-notes.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Done updating consultation-notes.html")
