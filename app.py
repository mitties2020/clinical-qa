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
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# -----------------------------------
# Whisper config (stable on Render)
# -----------------------------------
WHISPER_MODEL_SIZE = "tiny"   # IMPORTANT: tiny is reliable on free/low CPU

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

    tmp_webm = None
    tmp_wav = None

    try:
        f = request.files["audio"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as t1:
            tmp_webm = t1.name
            f.save(tmp_webm)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t2:
            tmp_wav = t2.name

        # 🔑 CRITICAL FIX: convert browser audio → WAV
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
            vad_filter=False,  # IMPORTANT: VAD kills dictation
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
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing API key"}), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an Australian medical clinical decision-support AI. "
                    "Respond using structured sections with clear headings: "
                    "Summary, Assessment, Differential Diagnosis, Investigations, "
                    "Management, Monitoring, Red Flags, References. "
                    "Base recommendations on Australian guidelines (ETG, AMH, "
                    "Therapeutic Guidelines, RACGP) where applicable."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "temperature": 0.2,
        "max_tokens": 1200
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = session.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=40)
        resp.raise_for_status()
        data = resp.json()

        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        return jsonify({"answer": answer})

    except Exception as e:
        print("DEEPSEEK ERROR:", e)
        return jsonify({"error": "AI request failed"}), 502

# -----------------------------------
# Local run
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
