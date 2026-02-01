import os
import time
import uuid
import tempfile
import threading
import requests

from flask import Flask, request, jsonify, render_template, redirect, session as flask_session
from faster_whisper import WhisperModel
import msal

# -----------------------------------
# Load .env ONLY locally (not Render)
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
# Microsoft / OneNote config
# -----------------------------------
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
MS_TENANT_ID = os.getenv("MS_TENANT_ID")
MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI")

ONENOTE_SECTION_ID = (os.getenv("ONENOTE_SECTION_ID") or "").strip()

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
AUTHORITY = f"https://login.microsoftonline.com/{MS_TENANT_ID}" if MS_TENANT_ID else None
SCOPES = ["User.Read", "Notes.Create"]

# -----------------------------------
# Flask app
# -----------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SESSION_SECRET", "dev_only_change_me")

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
# MSAL helpers
# -----------------------------------
def _msal_app():
    if not (MS_CLIENT_ID and MS_CLIENT_SECRET and MS_TENANT_ID and MS_REDIRECT_URI and AUTHORITY):
        return None
    return msal.ConfidentialClientApplication(
        MS_CLIENT_ID,
        authority=AUTHORITY,
        client_credential=MS_CLIENT_SECRET
    )

def _get_access_token():
    tok = flask_session.get("ms_token")
    if isinstance(tok, dict) and tok.get("access_token"):
        return tok["access_token"]
    return None

def _onenote_html_page(title: str, body_text: str) -> str:
    safe_title = (title or "WR Note").replace("&", "and").strip()
    safe_body = (body_text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return f"""<!DOCTYPE html>
<html>
<head>
  <title>{safe_title}</title>
  <meta name="created" content="{created}" />
</head>
<body>
  <h1>{safe_title}</h1>
  <pre>{safe_body}</pre>
</body>
</html>"""

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
        return jsonify({"error": "Missing audio field 'audio'."}), 400

    if not _transcribe_lock.acquire(blocking=False):
        return jsonify({"error": "Server busy. Try again."}), 429

    f = request.files["audio"]
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        whisper_model = get_whisper_model()

        segments, _ = whisper_model.transcribe(
            tmp_path,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        return jsonify({"text": text})

    except Exception as e:
        print("TRANSCRIBE ERROR:", repr(e))
        return jsonify({"error": "Transcription failed."}), 502

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
        return jsonify({
            "error": "Server misconfigured: missing DEEPSEEK_API_KEY"
        }), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided."}), 400

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are an Australian clinical education assistant."},
            {"role": "user", "content": query},
        ],
        "temperature": 0.25,
        "max_tokens": 1100,
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
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return jsonify({"answer": answer})

    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "DeepSeek request failed."}), 502

# ---------- AUTH ----------
@app.route("/auth/login")
def auth_login():
    app_obj = _msal_app()
    if not app_obj:
        return "Microsoft auth not configured.", 500

    state = str(uuid.uuid4())
    flask_session["ms_state"] = state

    auth_url = app_obj.get_authorization_request_url(
        scopes=SCOPES,
        state=state,
        redirect_uri=MS_REDIRECT_URI,
        prompt="select_account",
    )
    return redirect(auth_url)

@app.route("/auth/callback")
def auth_callback():
    app_obj = _msal_app()
    if not app_obj:
        return "Microsoft auth not configured.", 500

    if request.args.get("state") != flask_session.get("ms_state"):
        return "State mismatch", 400

    code = request.args.get("code")
    result = app_obj.acquire_token_by_authorization_code(
        code,
        scopes=SCOPES,
        redirect_uri=MS_REDIRECT_URI,
    )

    if "access_token" not in result:
        return "Auth failed", 400

    flask_session["ms_token"] = result
    return redirect("/")

# ---------- SEND TO ONENOTE ----------
@app.route("/api/onenote/send", methods=["POST"])
def onenote_send():
    token = _get_access_token()
    if not token:
        return jsonify({"error": "Not signed in."}), 401

    data = request.get_json(silent=True) or {}
    title = data.get("title", "WR Note")
    content = data.get("content", "")

    html = _onenote_html_page(title, content)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "text/html",
    }

    if ONENOTE_SECTION_ID:
        url = f"{GRAPH_BASE}/me/onenote/sections/{ONENOTE_SECTION_ID}/pages"
    else:
        url = f"{GRAPH_BASE}/me/onenote/pages?sectionName=EBM%20Notes"

    r = requests.post(url, headers=headers, data=html.encode("utf-8"), timeout=40)
    if r.status_code not in (200, 201):
        return jsonify({"error": "Graph API error"}), 502

    return jsonify({"ok": True})

# -----------------------------------
# Run locally
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
