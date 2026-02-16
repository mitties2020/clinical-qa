import os
import re
import tempfile
import threading
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, request, jsonify, render_template
from faster_whisper import WhisperModel

# -----------------------------------
# Load .env locally only (not Render)
# -----------------------------------
if os.getenv("RENDER") is None:
    from dotenv import load_dotenv
    load_dotenv()

# -----------------------------------
# DeepSeek config
# -----------------------------------
DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_MODEL = (os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
DEEPSEEK_URL = (os.getenv("DEEPSEEK_URL") or "https://api.deepseek.com/v1/chat/completions").strip()

# -----------------------------------
# Whisper config
# -----------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")

# -----------------------------------
# Flask app
# -----------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
http = requests.Session()

# -----------------------------------
# Whisper model (load once)
# -----------------------------------
_whisper_model = None
_whisper_init_lock = threading.Lock()
_transcribe_lock = threading.Lock()

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_init_lock:
            if _whisper_model is None:
                _whisper_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device="cpu",
                    compute_type="int8"
                )
    return _whisper_model

def awst_timestamp() -> str:
    dt = datetime.now(ZoneInfo("Australia/Perth"))
    return dt.strftime("%d %b %Y, %H:%M (AWST)")

def extract_field(text: str, labels: list[str]) -> str:
    if not text:
        return ""
    for lab in labels:
        pattern = rf"(?im)^\s*{re.escape(lab)}\s*[:=\-]\s*(.+?)\s*$"
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return ""

def normalise_card_type(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    if "gold" in s:
        return "Gold"
    if "white" in s:
        return "White"
    return s[:1].upper() + s[1:]

def build_dva_header(user_text: str) -> str:
    name = extract_field(user_text, ["DVA patient name", "Patient name", "Name", "Patient"])
    card = extract_field(user_text, ["DVA card", "Card type", "Card", "DVA card type"])
    dva_no = extract_field(user_text, ["DVA number", "DVA no", "File number", "File no"])
    accepted = extract_field(user_text, ["Accepted conditions", "Accepted condition", "Accepted", "White card accepted conditions"])
    referral = extract_field(user_text, ["Referral type", "Referral", "Requested referral", "Discipline"])
    contact = extract_field(user_text, ["Contact number", "Phone", "Mobile", "Contact"])

    card = normalise_card_type(card)

    header = []
    header.append(f"DVA Patient Name: {name or ''}")
    header.append(f"DVA Card Type: {card or 'Not specified'}")
    header.append(f"DVA Number: {dva_no or ''}")
    header.append(f"Accepted Conditions: {accepted or 'Not specified'}")
    header.append(f"Referral Type: {referral or ''}")
    header.append(f"Contact Number: {contact or ''}")
    header.append("")
    header.append("Telehealth Consult:")
    header.append("Dr Michael Addis")
    header.append(f"Date & Time (AWST): {awst_timestamp()}")
    return "\n".join(header).strip()

# -----------------------------------
# Prompts
# -----------------------------------
CLINICAL_SYSTEM_PROMPT = (
    "You are an Australian clinical education assistant for qualified medical doctors.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\nFollow-up & Safety Netting\nRed Flags\nReferences\n\n"
    "STYLE:\n"
    "Plain text only. Registrar-level depth. Australian practice framing.\n"
)

DVA_SYSTEM_PROMPT = (
    "You are an Australian medical practitioner assisting other qualified clinicians with DVA documentation.\n\n"
    "Primary use-case: DVA D0904 allied health referrals (new + renewal).\n\n"
    "IMPORTANT:\n"
    "Do not invent accepted conditions or entitlements. Do not advise misrepresentation.\n"
    "You may propose legitimate alternative pathways.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "DVA_META\n"
    "Referral type: <D0904 new | D0904 renewal | other/unclear>\n"
    "Provider type: <dietitian | physiotherapist | exercise physiologist | psychologist | OT | podiatrist | other/unclear>\n"
    "Provider-type checks:\n"
    "- <bullet>\n"
    "Renewal audit checks:\n"
    "- <bullet>\n"
    "Justification strength: <strong | moderate | weak>\n"
    "Audit risk: <low | medium | high>\n"
    "Missing items:\n"
    "- <bullet>\n"
    "Suggested amendments:\n"
    "- <bullet>\n"
    "Alternative legitimate pathways:\n"
    "- <bullet>\n"
    "END_DVA_META\n\n"
    "Then output clinical sections:\n"
    "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\nFollow-up & Safety Netting\nRed Flags\nReferences\n"
)

# -----------------------------------
# Routes
# -----------------------------------
@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def generate():
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    mode = (data.get("mode") or "clinical").strip().lower()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if mode.startswith("dva"):
        header = build_dva_header(query)
        referral_intent = "D0904 new" if mode == "dva_new" else "D0904 renewal" if mode == "dva_renew" else "D0904 (unspecified)"

        user_content = (
            f"Referral intent: {referral_intent}\n\n"
            f"{header}\n\n"
            f"DETAILS:\n{query}\n\n"
            "Follow DVA_META format then clinical headings."
        )
        system_prompt = DVA_SYSTEM_PROMPT
    else:
        user_content = f"Clinical question:\n{query}"
        system_prompt = CLINICAL_SYSTEM_PROMPT

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.25,
        "top_p": 0.9,
        "max_tokens": 1800,
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = http.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=70)
        resp.raise_for_status()
        out = resp.json()

        answer = (
            (out.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        return jsonify({"answer": answer or "No response."})

    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "AI request failed"}), 502

# -----------------------------------
# Local run
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
