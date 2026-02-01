import os
import tempfile
import threading
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
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# -----------------------------------
# Flask app
# -----------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
session = requests.Session()

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
# Routes
# -----------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ---------- TRANSCRIBE ----------
@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Missing audio"}), 400

    if not _transcribe_lock.acquire(blocking=False):
        return jsonify({"error": "Server busy"}), 429

    tmp_path = None
    try:
        f = request.files["audio"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        model = get_whisper_model()
        segments, _ = model.transcribe(
            tmp_path,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            temperature=0.0,
            condition_on_previous_text=False,
        )

        text = " ".join(s.text.strip() for s in segments).strip()
        return jsonify({"text": text})

    except Exception as e:
        print("TRANSCRIBE ERROR:", e)
        return jsonify({"error": "Transcription failed"}), 502

    finally:
        _transcribe_lock.release()
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# ---------- GENERATE ----------
@app.route("/api/generate", methods=["POST"])
def generate():
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing API key"}), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Doctor-level, Australian-context, sectioned plain-text output for your UI parser.
    system_prompt = (
        "You are an Australian clinical education assistant for qualified medical doctors.\n\n"

        "ROLE AND CONTEXT:\n"
        "This system is used by Australian hospital doctors (PGY2+, registrars, consultants).\n"
        "The purpose is clinical reasoning, safe bedside understanding, and exam-style revision.\n"
        "This is not patient-facing health education.\n\n"

        "OUTPUT FORMAT (MANDATORY):\n"
        "You MUST structure every response using the following headings EXACTLY, each on its own line, in this order:\n"
        "Summary\n"
        "Assessment\n"
        "Diagnosis\n"
        "Investigations\n"
        "Treatment\n"
        "Monitoring\n"
        "Follow-up & Safety Netting\n"
        "Red Flags\n"
        "References\n\n"

        "FORMAT RULES:\n"
        "Use plain text only.\n"
        "Do not use markdown symbols such as ###, **, *, -, or •.\n"
        "Do not collapse the response into one paragraph.\n"
        "Under each heading, write multiple short, clinically meaningful lines or paragraphs.\n"
        "You may use separate lines to convey lists, but without bullet characters.\n"
        "If a section is not relevant, write: Not applicable.\n\n"

        "CLINICAL STANDARDS:\n"
        "Align content with contemporary Australian practice.\n"
        "Do not quote proprietary resources verbatim (e.g. ETG/AMH).\n"
        "If discussing medicines, provide general approach and typical adult dose ranges only; advise checking local references.\n"
        "For paediatrics, pregnancy, renal/hepatic impairment, and drug interactions, flag considerations and advise local checking.\n\n"

        "DEPTH REQUIREMENT:\n"
        "Aim for registrar-level depth by default.\n"
        "Avoid superficial public-health style explanations.\n"
    )

    user_content = (
        "This is a hypothetical, de-identified clinical question for educational purposes.\n\n"
        f"Clinical question:\n{query}"
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.25,
        "top_p": 0.9,
        "max_tokens": 1400,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = session.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        answer = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        # Post-clean: if model sneaks in markdown, strip common markers safely.
        if answer:
            answer = answer.replace("\r", "")
            answer = answer.replace("###", "")
            answer = answer.replace("**", "")

            cleaned_lines = []
            for line in answer.splitlines():
                # Strip bullet chars at line start (if any)
                cleaned_lines.append(line.lstrip("-•* ").rstrip())
            answer = "\n".join(cleaned_lines).strip()

        return jsonify({"answer": answer or "No response."})

    except Exception as e:
        print("DEEPSEEK ERROR:", e)
        return jsonify({"error": "AI request failed"}), 502

# -----------------------------------
# Local run
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
