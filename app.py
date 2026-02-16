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
# tiny is most reliable on low-CPU instances
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
    # Western Australia time
    dt = datetime.now(ZoneInfo("Australia/Perth"))
    return dt.strftime("%d %b %Y, %H:%M (AWST)")

def extract_field(text: str, labels: list[str]) -> str:
    """
    Extracts value after labels like:
    'DVA Patient Name: John Smith'
    'Name - John Smith'
    'Name=John Smith'
    """
    if not text:
        return ""
    for lab in labels:
        # label followed by : or - or =
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
    "Audience:\n"
    "Australian hospital/GP doctors (PGY2+, registrars, consultants). Not patient-facing.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "Use these headings EXACTLY, each on its own line:\n"
    "Summary\n"
    "Assessment\n"
    "Diagnosis\n"
    "Investigations\n"
    "Treatment\n"
    "Monitoring\n"
    "Follow-up & Safety Netting\n"
    "Red Flags\n"
    "References\n\n"
    "STYLE:\n"
    "Plain text only. No markdown symbols (###, **, *, •).\n"
    "Registrar-level depth. Australian practice framing.\n"
    "Do not quote proprietary sources verbatim.\n"
    "References should name source families only (e.g. Therapeutic Guidelines/eTG, AMH, Australian Immunisation Handbook, local protocols).\n"
)

# Updated DVA prompt: includes DVA_META block + provider checks + renewal audit checks.
DVA_SYSTEM_PROMPT = (
    "You are an Australian medical practitioner assisting other qualified clinicians with DVA documentation.\n\n"
    "Primary use-case:\n"
    "DVA D0904 allied health referrals (new + renewal).\n\n"
    "Goal:\n"
    "Assess whether the referral justification is defensible and identify missing elements that increase audit risk.\n"
    "Provide clinician-grade suggestions to strengthen documentation and explore legitimate alternative pathways.\n"
    "Do not claim DVA approval.\n\n"
    "IMPORTANT:\n"
    "Do not invent accepted conditions, entitlements, or facts not provided.\n"
    "If details are missing, explicitly say what is missing and why it matters.\n"
    "You may propose LEGITIMATE alternatives (e.g., request clarification, GP review, specialist input, different discipline, IL pathway) "
    "but do not advise gaming eligibility or misrepresenting linkage.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "First output a machine-readable block exactly like this:\n"
    "DVA_META\n"
    "Referral type: <D0904 new | D0904 renewal | other/unclear>\n"
    "Provider type: <dietitian | physiotherapist | exercise physiologist | psychologist | OT | podiatrist | other/unclear>\n"
    "Provider-type checks:\n"
    "- <bullet>\n"
    "- <bullet>\n"
    "Renewal audit checks:\n"
    "- <bullet>\n"
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
    "Then output the clinical answer with these headings EXACTLY, each on its own line:\n"
    "Summary\n"
    "Assessment\n"
    "Diagnosis\n"
    "Investigations\n"
    "Treatment\n"
    "Monitoring\n"
    "Follow-up & Safety Netting\n"
    "Red Flags\n"
    "References\n\n"
    "Within the sections:\n"
    "Assessment must include: justification strength + gaps + audit-risk flags.\n"
    "Treatment must include: a rewritten, individualised clinical note paragraph suitable for records.\n"
    "Style: Plain text only. No markdown.\n"
)

# -----------------------------------
# Routes
# -----------------------------------
@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception:
        return "vividmedi backend running", 200

# ---------- TRANSCRIBE ----------
@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """
    Browser uploads audio/webm (opus).
    Convert to 16kHz mono WAV with ffmpeg, then transcribe with faster_whisper.
    """
    if "audio" not in request.files:
        return jsonify({"error": "Missing audio"}), 400

    if not _transcribe_lock.acquire(blocking=False):
        return jsonify({"error": "Server busy"}), 429

    tmp_webm = None
    tmp_wav = None

    try:
        f = request.files["audio"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as t1:
            tmp_webm = t1.name
            f.save(tmp_webm)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t2:
            tmp_wav = t2.name

        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_webm, "-ac", "1", "-ar", "16000", tmp_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        model = get_whisper_model()
        segments, _ = model.transcribe(
            tmp_wav,
            language="en",
            vad_filter=False,
            temperature=0.0,
            beam_size=5,
            condition_on_previous_text=False,
        )

        text = " ".join(s.text.strip() for s in segments).strip()
        return jsonify({"text": text})

    except Exception as e:
        print("TRANSCRIBE ERROR:", repr(e))
        return jsonify({"error": "Transcription failed"}), 502

    finally:
        _transcribe_lock.release()
        for p in (tmp_webm, tmp_wav):
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass

# ---------- GENERATE ----------
@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Request JSON:
      { "query": "...", "mode": "clinical" | "dva_new" | "dva_renew" | "dva" }

    - mode defaults to "clinical"
    """
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    mode = (data.get("mode") or "clinical").strip().lower()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Accept legacy "dva" and new "dva_*" modes
    if mode.startswith("dva"):
        header = build_dva_header(query)

        if mode == "dva_renew":
            referral_intent = "D0904 renewal"
        elif mode == "dva_new":
            referral_intent = "D0904 new"
        else:
            referral_intent = "D0904 (new/renewal not specified)"

        user_content = (
            "You are assessing a DVA allied health referral. Focus on D0904 logic.\n"
            f"Referral intent (UI-selected): {referral_intent}\n\n"
            "PATIENT HEADER (must appear at the top of your response exactly as provided):\n"
            f"{header}\n\n"
            "DETAILS (verbatim paste from clinician/UI):\n"
            f"{query}\n\n"
            "Instructions:\n"
            "1) Output DVA_META block first (exact format required).\n"
            "2) Provider-type checks must be discipline-specific (dietitian/physio/EP/psych/OT/podiatry) and include typical DVA admin pitfalls.\n"
            "3) Renewal audit checks must be strict: measurable benefit, ongoing impairment, unmet goals, sessions used/remaining, "
            "and whether an End-of-Cycle report content is present.\n"
            "4) Strengthen the documentation only with legitimate strategies (clarify linkage, better clinical reasoning, alternative pathways). "
            "Do NOT advise misrepresentation or gaming eligibility.\n"
            "5) Then provide the structured headings output.\n"
            "6) Keep it Australian clinician-facing.\n"
        )

        system_prompt = DVA_SYSTEM_PROMPT
    else:
        user_content = (
            "This is a hypothetical, de-identified clinical question for educational purposes.\n\n"
            f"Clinical question:\n{query}"
        )
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
        "stream": False,
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

        # Light cleanup if model sneaks markdown in
        if answer:
            answer = answer.replace("\r", "")
            answer = answer.replace("###", "")
            answer = answer.replace("**", "")
            cleaned = []
            for line in answer.splitlines():
                cleaned.append(line.lstrip("•*- ").rstrip())
            answer = "\n".join(cleaned).strip()

        return jsonify({"answer": answer or "No response."})

    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "AI request failed"}), 502

# -----------------------------------
# Local run
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
