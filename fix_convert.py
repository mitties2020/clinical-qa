#!/usr/bin/env python3
import re

with open('app.py', 'r') as f:
    content = f.read()

# Find the problematic convert_notes function and replace it cleanly
pattern = r'@app\.post\("/convert-notes"\).*?return jsonify\(\{"error": "Conversion failed"\}\), 502'

replacement = '''@app.post("/convert-notes")
def convert_notes():
    """Convert clinical data into clinical notes with type-specific formatting"""
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    clinical_data = (data.get("clinical_data") or "").strip()
    note_type = (data.get("note_type") or "consultation_note").strip().lower()

    if not clinical_data:
        return jsonify({"error": "Empty clinical data"}), 400
    
    if len(clinical_data) < MIN_QUERY_LENGTH:
        return jsonify({"error": f"Input too short (min {MIN_QUERY_LENGTH} chars)"}), 400
    if len(clinical_data) > MAX_QUERY_LENGTH:
        return jsonify({"error": f"Input too long (max {MAX_QUERY_LENGTH} chars)"}), 400

    blocked = enforce_quota_or_402()
    if blocked:
        return blocked

    try:
        u = get_authed_user()
        user_id = u["id"] if u else None
        context = get_conversation_context(user_id) if user_id else ""

        note_prompts = {
            "consultation_note": "Generate a CONSULTATION NOTE with: DATE, PRESENTING COMPLAINT, HPI, PMHx, MEDICATIONS, ALLERGIES, SOCIAL HISTORY, EXAMINATION, ASSESSMENT, PLAN, FOLLOW-UP. Be concise and professional.",
            "referral_letter": "Generate a formal REFERRAL LETTER with: DATE, RECIPIENT, RE, PATIENT DETAILS, REASON FOR REFERRAL, CLINICAL HISTORY, MANAGEMENT, INVESTIGATIONS. Make it professional.",
            "discharge_summary": "Generate a DISCHARGE SUMMARY with: DATES, DIAGNOSES, PROCEDURES, HOSPITAL COURSE, FINAL DIAGNOSIS, MEDICATIONS, FOLLOW-UP. Be clear and complete.",
            "progress_note": "Generate a PROGRESS NOTE in SOAP format: SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN.",
            "clinical_report": "Generate a CLINICAL REPORT with: TITLE, DATE, PATIENT INFO, FINDINGS, INTERPRETATION, CONCLUSION.",
            "soap_note": "Generate a SOAP NOTE: S (history), O (vitals/exam), A (diagnosis), P (management).",
            "procedure_note": "Generate a PROCEDURE NOTE: NAME, DATE, INDICATION, TECHNIQUE, FINDINGS, COMPLICATIONS, ORDERS."
        }

        system_prompt = note_prompts.get(note_type, note_prompts["consultation_note"])
        system_prompt = "You are an expert Australian clinical scribe. " + system_prompt + " Output ONLY the report. Format for medical records."
        user_content = "Generate from: " + clinical_data

        answer = call_deepseek(system_prompt, user_content, context)

        if user_id:
            save_conversation(user_id, clinical_data[:100], answer, f"notes_{note_type}")

        return jsonify({"clinical_notes": answer})

    except Exception as e:
        print("CONVERT NOTES ERROR:", repr(e))
        return jsonify({"error": "Conversion failed"}), 502'''

content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open('app.py', 'w') as f:
    f.write(content)

print("Replaced convert_notes function")
