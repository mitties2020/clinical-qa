import os
import tempfile
import threading
import subprocess
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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


# -----------------------------------
# Whisper config
# -----------------------------------
# "tiny" is far more reliable on free/low-CPU instances.
# You can change to "base" later if you're on paid Render and want slightly higher accuracy.
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


# -----------------------------------
# Prompts
# -----------------------------------
CLINICAL_SYSTEM_PROMPT = (
    "You are an Australian clinical education assistant for qualified medical doctors.\n\n"
    "Audience and intent:\n"
    "This is for Australian hospital/GP doctors (PGY2+, registrars, consultants).\n"
    "This is not patient-facing education.\n\n"

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

    "STYLE RULES:\n"
    "Use plain text. Avoid markdown symbols (###, **, *, •).\n"
    "Do not collapse into one paragraph.\n"
    "Write at registrar-level depth.\n"
    "Be clinically practical and Australian-context.\n"
    "Do not quote proprietary sources verbatim.\n"
    "References should be general source families only (e.g. Therapeutic Guidelines/eTG, AMH, Australian Immunisation Handbook, local protocols).\n"
)

DVA_SYSTEM_PROMPT = (
    "You are an Australian medical practitioner assisting other qualified clinicians with DVA-funded care documentation.\n\n"
    "Purpose:\n"
    "Assess whether the provided details support a defensible DVA referral request and generate an individualised clinical note.\n"
    "This is documentation support, not official approval. Do not claim DVA has approved anything.\n\n"

    "Key constraints:\n"
    "Do not invent accepted conditions.\n"
    "If card type or accepted conditions are missing/unclear, say so.\n"
    "Do not quote proprietary resources verbatim.\n"
    "Use Australian spelling.\n"
    "Make the note sound genuinely patient-specific (vary sentence structure, use the provided details, avoid boilerplate).\n\n"

    "OUTPUT FORMAT (MANDATORY):\n"
    "Use these headings EXACTLY, each on its own line:\n"
    "Patient & DVA Context\n"
    "Clinical Summary\n"
    "Referral Assessment\n"
    "Missing or Weak Elements\n"
    "Audit Risk Flags\n"
    "Suggested Improved Clinical Note\n"
    "Next Steps Checklist\n"
    "References\n\n"

    "Guidance for reasoning:\n"
    "Link the referral to functional impact and to the most likely accepted condition where applicable.\n"
    "Flag when the referral looks non-specific, disproportionate, or poorly linked to an accepted condition.\n"
    "If information is insufficient, explicitly list what is required to make it defensible.\n"
    "References should mention general Australian/DVA guidance families only (e.g. DVA allied health referral guidance, MBS item notes, local documentation standards).\n"
)


# -----------------------------------
# Routes
# -----------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------- TRANSCRIBE ----------
@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """
    Browser uploads audio/webm (opus). Convert to 16kHz mono wav with ffmpeg, then transcribe.
    This is the most reliable approach on Render.
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

        # Convert webm → wav (16kHz mono)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", tmp_webm,
                "-ac", "1",
                "-ar", "16000",
                tmp_wav,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        model = get_whisper_model()
        segments, _ = model.transcribe(
            tmp_wav,
            language="en",
            vad_filter=False,               # VAD often deletes dictation segments
            temperature=0.0,
            beam_size=5,
            condition_on_previous_text=False,
        )

        text = " ".join(s.text.strip() for s in segments).strip()
        return jsonify({"text": text})

    except Exception as e:
        # Keep logs server-side
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
      { "query": "...", "mode": "clinical" | "dva" }

    - mode defaults to "clinical"
    """
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    mode = (data.get("mode") or "clinical").strip().lower()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if mode == "dva":
        system_prompt = DVA_SYSTEM_PROMPT
        user_content = (
            "Use the details below to assess DVA referral justification and write an individualised clinical note.\n\n"
            "INPUT:\n"
            f"{query}\n\n"
            "Remember: do not fabricate accepted conditions. If missing, list what is required."
        )
    else:
        system_prompt = CLINICAL_SYSTEM_PROMPT
        user_content = (
            "This is a hypothetical, de-identified clinical question for educational purposes.\n\n"
            f"Clinical question:\n{query}"
        )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.25,
        "top_p": 0.9,
        "max_tokens": 1400,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = http.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        out = resp.json()

        answer = (
            (out.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        # Light cleanup to keep your UI parser happy if model sneaks markdown in.
        if answer:
            answer = answer.replace("\r", "")
            answer = answer.replace("###", "")
            answer = answer.replace("**", "")
            cleaned_lines = []
            for line in answer.splitlines():
                cleaned_lines.append(line.lstrip("•*- ").rstrip())
            answer = "\n".join(cleaned_lines).strip()

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
