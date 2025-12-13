import os
import tempfile
import threading
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny | base | small

if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# --------------------------------------------------
# APP
# --------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
http = requests.Session()

# --------------------------------------------------
# WHISPER (LOAD ONCE)
# --------------------------------------------------
_whisper_model = None
_whisper_lock = threading.Lock()
_transcribe_lock = threading.Lock()

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                print(f"[INIT] Loading Whisper model: {WHISPER_MODEL_SIZE}")
                _whisper_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device="cpu",
                    compute_type="int8"
                )
    return _whisper_model

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# --------------------------------------------------
# TRANSCRIBE (STABLE, MEDICAL-BIASED)
# --------------------------------------------------
@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "Missing audio"}), 400

    if not _transcribe_lock.acquire(blocking=False):
        return jsonify({"error": "Server busy"}), 429

    tmp_path = None
    try:
        audio = request.files["audio"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp_path = tmp.name
            audio.save(tmp_path)

        model = get_whisper_model()

        medical_prompt = (
            "Australian clinical dictation. Common medical terms: "
            "morphine, fentanyl, ketamine, ondansetron, metoclopramide, "
            "paracetamol, ibuprofen, ceftriaxone, amoxicillin, adrenaline, "
            "noradrenaline, anaphylaxis, seizure, asthma, COPD, DKA, stroke, "
            "atrial fibrillation, dose, dosage, milligrams, micrograms, per kilogram."
        )

        segments, info = model.transcribe(
            tmp_path,
            language="en",
            initial_prompt=medical_prompt,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False
        )

        raw = " ".join(seg.text.strip() for seg in segments).strip()

        # ---- Post-correction (high-yield fixes) ----
        fixes = {
            "morphine dozer": "morphine dose",
            "morphine dosage": "morphine dose",
            "on dansetron": "ondansetron",
            "ondan setron": "ondansetron",
            "paracetemol": "paracetamol",
            "meto clopramide": "metoclopramide",
            "milligram": "milligrams",
            "microgram": "micrograms",
        }

        text = raw.lower()
        for k, v in fixes.items():
            text = text.replace(k, v)

        return jsonify({
            "text": text.strip(),
            "raw": raw,
            "language": getattr(info, "language", "en")
        })

    except Exception as e:
        print("[TRANSCRIBE ERROR]", repr(e))
        return jsonify({"error": "Transcription failed"}), 502

    finally:
        _transcribe_lock.release()
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# --------------------------------------------------
# GENERATE (DEEPSEEK)
# --------------------------------------------------
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "No query provided"}), 400

    system_prompt = (
        "You are an AI clinical education assistant for Australian clinicians.\n"
        "This is educational only, not real-time patient care.\n"
        "Use Australian spelling and structured headings.\n"
        "Do not issue directives. Encourage checking local guidelines.\n\n"
        "Structure responses using:\n"
        "Summary\nAssessment\nDiagnosis\nInvestigations\n"
        "Treatment\nMonitoring\nFollow-up & Safety Netting\n"
        "Red Flags\nReferences\n"
    )

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "temperature": 0.25,
        "max_tokens": 1100
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = http.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=40)
        r.raise_for_status()
        data = r.json()

        answer = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            return jsonify({"error": "Empty response"}), 502

        return jsonify({"answer": answer})

    except Exception as e:
        print("[DEEPSEEK ERROR]", repr(e))
        return jsonify({"error": "LLM request failed"}), 502

# --------------------------------------------------
# MAIN (DEV ONLY)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
