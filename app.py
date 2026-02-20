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
# Health check (Render)
# IMPORTANT: define ONCE only (duplicate routes break gunicorn)
# -----------------------------------
@app.get("/healthz")
def healthz():
    return "ok", 200

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

# -----------------------------------
# Helpers
# -----------------------------------
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

def _run_ffmpeg_to_wav(in_path: str, out_path: str) -> None:
    """
    Convert whatever the browser recorded (webm/ogg/mp3/etc) to a clean 16kHz mono wav
    for Whisper. Requires ffmpeg installed on Render (you already apt-get it).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", in_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        out_path,
    ]
    # Capture stderr so we can diagnose failures.
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + (p.stderr.decode("utf-8", "ignore")[-800:] or "unknown error"))

def _clean_transcript(text: str) -> str:
    # Keep light-touch; you already do corrections client-side.
    t = (text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# -----------------------------------
# Prompts (keep headings consistent with your front-end accordion)
# -----------------------------------
CLINICAL_SYSTEM_PROMPT = (
    "You are an Australian clinical education assistant for qualified medical doctors.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\nFollow-up & Safety Netting\nRed Flags\nReferences\n\n"
    "STYLE:\n"
    "Plain text only. Registrar-level depth. Australian practice framing.\n"
    "If the user pastes raw data (old notes, imaging, labs), extract and reformat cleanly.\n"
)

HANDOVER_SYSTEM_PROMPT = (
    "You are an Australian ED-focused clinical assistant.\n"
    "Task: produce a high-quality HANDOVER / PRESENTATION that suits the most likely context.\n"
    "Infer context (ED, ward, ICU, clinic) from the content and tailor accordingly.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\nFollow-up & Safety Netting\nRed Flags\nReferences\n\n"
    "CONTENT RULES:\n"
    "- Summary should be an ISBAR-style handover (concise, immediately usable).\n"
    "- Include key vitals/status, working diagnosis, key differentials, what has been done, and what's pending.\n"
    "- If pasted data is messy, reorganise into a clean handover.\n"
)

CLINICAL_NOTE_SYSTEM_PROMPT = (
    "You are an Australian clinical documentation assistant.\n"
    "Task: turn the provided text/dictation/pasted results into a structured clinical note.\n"
    "Use Australian spelling. Avoid fabricating details; use placeholders when unknown.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\nFollow-up & Safety Netting\nRed Flags\nReferences\n\n"
    "CONTENT RULES:\n"
    "- Summary: structured note skeleton (presenting complaint, HPI, relevant PMHx/meds/allergies, exam, impression, plan).\n"
    "- Use concise, clinically realistic phrasing appropriate for ED/ward notes.\n"
    "- If the user pastes labs/imaging/history, format into readable sections.\n"
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
@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/generate")
def generate():
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    mode = (data.get("mode") or "clinical").strip().lower()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Choose prompt by mode
    if mode.startswith("dva"):
        header = build_dva_header(query)
        referral_intent = (
            "D0904 new" if mode == "dva_new"
            else "D0904 renewal" if mode == "dva_renew"
            else "D0904 (unspecified)"
        )
        user_content = (
            f"Referral intent: {referral_intent}\n\n"
            f"{header}\n\n"
            f"DETAILS:\n{query}\n\n"
            "Follow DVA_META format then clinical headings."
        )
        system_prompt = DVA_SYSTEM_PROMPT

    elif mode in ("handover", "handover_summary"):
        user_content = (
            "Create a clinician-to-clinician handover/presentation from the following content.\n"
            "If context is unclear, infer the most likely.\n\n"
            f"{query}"
        )
        system_prompt = HANDOVER_SYSTEM_PROMPT

    elif mode in ("clinical_note", "note", "create_clinical_note"):
        user_content = (
            "Create a structured clinical note from the following content (dictation / pasted data).\n"
            "Do not invent details; use placeholders when missing.\n\n"
            f"{query}"
        )
        system_prompt = CLINICAL_NOTE_SYSTEM_PROMPT

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

@app.post("/api/transcribe")
def transcribe():
    """
    Browser uploads audio as FormData 'audio' (webm/ogg).
    We convert -> wav via ffmpeg then run faster-whisper.

    This route is REQUIRED for your Dictate buttons to work.
    """
    if "audio" not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    f = request.files["audio"]
    if not f or f.filename is None:
        return jsonify({"error": "Invalid audio upload"}), 400

    # Serialise transcription to prevent CPU thrash on tiny Render instances
    with _transcribe_lock:
        tmp_in = None
        tmp_wav = None
        try:
            # Save incoming blob
            suffix = os.path.splitext(f.filename)[1] or ".webm"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tin:
                tmp_in = tin.name
                f.save(tmp_in)

            # Convert to wav for whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tw:
                tmp_wav = tw.name

            _run_ffmpeg_to_wav(tmp_in, tmp_wav)

            model = get_whisper_model()

            # Higher accuracy knobs without going too slow:
            segments, info = model.transcribe(
                tmp_wav,
                language="en",
                vad_filter=True,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False,
            )

            text_parts = []
            for seg in segments:
                if seg and seg.text:
                    text_parts.append(seg.text.strip())

            text = _clean_transcript(" ".join(text_parts))
            return jsonify({"text": text})

        except Exception as e:
            print("TRANSCRIBE ERROR:", repr(e))
            return jsonify({"error": "Transcription failed"}), 500

        finally:
            for p in (tmp_in, tmp_wav):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

# -----------------------------------
# Local run
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
