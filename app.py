import os
import time
import uuid
import json
import tempfile
import threading
import requests

from flask import Flask, request, jsonify, render_template, redirect, session as flask_session
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# NEW: Microsoft auth
import msal

load_dotenv()

# -------------------------
# DeepSeek config (unchanged)
# -------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# -------------------------
# Whisper config (unchanged)
# -------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# -------------------------
# NEW: OneNote / Microsoft Graph config
# -------------------------
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
MS_TENANT_ID = os.getenv("MS_TENANT_ID")
MS_REDIRECT_URI = os.getenv("MS_REDIRECT_URI")  # e.g. https://www.clineraclinic.com/

# Optional: force a section (recommended once you know it)
ONENOTE_SECTION_ID = (os.getenv("ONENOTE_SECTION_ID") or "").strip()

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
AUTHORITY = f"https://login.microsoftonline.com/{MS_TENANT_ID}" if MS_TENANT_ID else None

# Keep least privilege to create pages
SCOPES = ["User.Read", "Notes.Create"]

if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("SESSION_SECRET", "dev_only_change_me")  # REQUIRED for login session

session = requests.Session()

# -------------------------
# Whisper model load-once + locks (prevents memory spikes)
# -------------------------
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

# -------------------------
# NEW: MSAL helpers
# -------------------------
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
    # OneNote create-page expects HTML request body.
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

# -------------------------
# Routes (existing)
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    """
    Receives audio as multipart/form-data with field name 'audio'
    Returns JSON: { "text": "..." }
    """
    if "audio" not in request.files:
        return jsonify({"error": "Missing audio field 'audio'."}), 400

    # Prevent overlapping transcribes (CPU/memory protection)
    if not _transcribe_lock.acquire(blocking=False):
        return jsonify({"error": "Server busy. Try again."}), 429

    f = request.files["audio"]
    if not f:
        _transcribe_lock.release()
        return jsonify({"error": "Empty audio upload."}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        whisper_model = get_whisper_model()

        # Optional prompt helps medical vocab a bit
        medical_prompt = (
            "Australian clinician dictation. Common terms: morphine, fentanyl, ketamine, ondansetron, "
            "metoclopramide, lamotrigine, levetiracetam, valproate, phenytoin, thiamine, Wernicke, "
            "re-feeding syndrome, phosphate, magnesium, NG tube, dosage, milligrams, micrograms, per kilogram."
        )

        segments, info = whisper_model.transcribe(
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

        text = " ".join(seg.text.strip() for seg in segments).strip()
        return jsonify({"text": text})

    except Exception as e:
        print("TRANSCRIBE ERROR:", repr(e))
        return jsonify({"error": "Transcription failed on server."}), 502

    finally:
        _transcribe_lock.release()
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided."}), 400

    system_prompt = (
        "You are an AI clinical education assistant for qualified clinicians and doctors "
        "in training working in Australian hospitals.\n\n"
        "Purpose and limits:\n"
        "- Your role is to support STUDY, REVISION and exam-style reasoning.\n"
        "- You are NOT providing live clinical decision support for real patients.\n"
        "- Do not present recommendations as orders or directives; instead frame them as "
        "educational guidance that must be checked against local protocols and senior advice.\n\n"
        "Jurisdiction and practice context:\n"
        "- Assume an Australian hospital setting unless clearly stated otherwise.\n"
        "- Bias your explanations toward what is broadly consistent with contemporary Australian "
        "practice and guidelines.\n"
        "- Do NOT name or quote proprietary resources.\n"
        "- Use Australian spelling.\n\n"
        "Safety and prescribing:\n"
        "- When you mention medicines, discuss broad dose ranges only and recommend checking local references.\n\n"
        "Response structure:\n"
        "Use plain-text headings (only include relevant ones):\n"
        "Summary\n"
        "Assessment\n"
        "Diagnosis\n"
        "Investigations\n"
        "Treatment\n"
        "Monitoring\n"
        "Follow-up & Safety Netting\n"
        "Red Flags\n"
        "References\n\n"
        "Formatting:\n"
        "- One key point per line.\n"
        "- No markdown symbols.\n"
    )

    user_content = (
        "This is a hypothetical, de-identified clinical study question for educational purposes only.\n\n"
        f"Clinical question:\n{query}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.25,
        "top_p": 0.9,
        "max_tokens": 1100,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = session.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=40)
        print("DeepSeek status:", resp.status_code)
        resp.raise_for_status()
        data = resp.json()
        answer = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not answer:
            return jsonify({"error": "Empty response from model."}), 502
        return jsonify({"answer": answer})
    except Exception as e:
        print("DeepSeek API error:", repr(e))
        return jsonify({"error": "Error contacting DeepSeek API."}), 502


# -------------------------
# NEW: OneNote Sign-in (adds routes; does not affect existing site)
# -------------------------
@app.route("/auth/login")
def auth_login():
    app_obj = _msal_app()
    if not app_obj:
        return (
            "Microsoft auth not configured. Set MS_CLIENT_ID, MS_CLIENT_SECRET, MS_TENANT_ID, MS_REDIRECT_URI, SESSION_SECRET.",
            500,
        )

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
    if not code:
        return "Missing auth code", 400

    result = app_obj.acquire_token_by_authorization_code(
        code,
        scopes=SCOPES,
        redirect_uri=MS_REDIRECT_URI,
    )

    if "access_token" not in result:
        return f"Auth failed: {result.get('error_description')}", 400

    flask_session["ms_token"] = result
    return redirect("/")


# -------------------------
# NEW: Send note to OneNote (backend API; UI button can call this later)
# -------------------------
@app.route("/api/onenote/send", methods=["POST"])
def onenote_send():
    token = _get_access_token()
    if not token:
        return jsonify({"error": "Not signed in. Visit /auth/login first."}), 401

    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "WR Note").strip()
    content = (data.get("content") or "").strip()

    if not content:
        return jsonify({"error": "Empty note content."}), 400

    html = _onenote_html_page(title, content)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "text/html",
    }

    # If you set ONENOTE_SECTION_ID, it'll always write there.
    if ONENOTE_SECTION_ID:
        url = f"{GRAPH_BASE}/me/onenote/sections/{ONENOTE_SECTION_ID}/pages"
    else:
        # Writes into a section named "EBM Notes" in your default notebook.
        url = f"{GRAPH_BASE}/me/onenote/pages?sectionName=EBM%20Notes"

    try:
        r = requests.post(url, headers=headers, data=html.encode("utf-8"), timeout=40)
        if r.status_code not in (200, 201):
            return jsonify({"error": "Graph error", "status": r.status_code, "detail": r.text}), 502
        return jsonify({"ok": True})
    except Exception as e:
        print("ONENOTE SEND ERROR:", repr(e))
        return jsonify({"error": "Failed to contact Microsoft Graph"}), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
