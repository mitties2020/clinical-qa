import os
import re
import time
import tempfile
import threading
import subprocess
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
from functools import wraps
from urllib.parse import urljoin

import requests
from flask import (
    Flask,
    g,
    request,
    jsonify,
    render_template,
    session,
    make_response,
    redirect,
    url_for,
)

from faster_whisper import WhisperModel
from performance_monitor import monitor

if os.getenv("RENDER") is None:
    from dotenv import load_dotenv
    load_dotenv()

DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_MODEL = (os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
DEEPSEEK_URL = (os.getenv("DEEPSEEK_URL") or "https://api.deepseek.com/v1/chat/completions").strip()

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
AUTH_CODE = (os.getenv("AUTH_CODE") or "").strip()
DB_PATH = os.getenv("DB_PATH") or "vividmedi.db"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "dev-insecure-change-me"

http = requests.Session()


@app.before_request
def _track_request_start():
    g._req_start = time.time()


@app.after_request
def _track_request_end(response):
    start = getattr(g, "_req_start", None)
    if start is not None:
        duration_ms = (time.time() - start) * 1000
        monitor.record_endpoint(request.path, request.method, response.status_code, duration_ms)
    return response


def require_auth(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if session.get("authenticated") is not True:
            if request.path.startswith("/api/") or request.path in {"/ask", "/convert-notes"}:
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped


def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_history_db():
    with db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_key TEXT NOT NULL,
                item_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def delete_history_entry(entry_id: int) -> bool:
    with db_conn() as conn:
        cur = conn.execute(
            "DELETE FROM history_entries WHERE id = ? AND user_key = ?",
            (entry_id, session_user_key()),
        )
        conn.commit()
        return cur.rowcount > 0


def clear_history_entries() -> int:
    with db_conn() as conn:
        cur = conn.execute("DELETE FROM history_entries WHERE user_key = ?", (session_user_key(),))
        conn.commit()
        return cur.rowcount
def session_user_key() -> str:
    return "code_user" if session.get("authenticated") is True else "guest"


def save_history(item_type: str, content: str):
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO history_entries (user_key, item_type, content, created_at) VALUES (?, ?, ?, ?)",
            (session_user_key(), item_type, content, datetime.utcnow().isoformat()),
        )
        conn.commit()


def load_history(limit: int = 200):
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT id, item_type, content, created_at FROM history_entries WHERE user_key = ? ORDER BY id DESC LIMIT ?",
            (session_user_key(), limit),
        ).fetchall()
    return [dict(r) for r in rows]


init_history_db()

@app.get("/health")
def health():
    return "ok", 200

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/_ping")
def ping():
    return "pong", 200


@app.post("/api/call-patient")
@require_auth
def call_patient():
    account_sid = (os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
    auth_token = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    twilio_number = (os.getenv("TWILIO_NUMBER") or "").strip()
    base_url = (os.getenv("BASE_URL") or request.host_url).strip()

    if not (account_sid and auth_token and twilio_number):
        return jsonify({
            "ok": False,
            "error": "Calling is not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_NUMBER."
        }), 503

    payload = request.get_json(silent=True) or {}
    patient_phone_raw = str(payload.get("patientPhone") or "").strip()
    patient_phone = normalize_au_phone(patient_phone_raw)
    if not patient_phone:
        return jsonify({"ok": False, "error": "Invalid patientPhone. Use E.164 (e.g. +614XXXXXXXX) or AU mobile format."}), 400

    base_url_norm = base_url if base_url.endswith("/") else f"{base_url}/"
    status_url = urljoin(base_url if base_url.endswith("/") else f"{base_url}/", "api/call-status")
    conference_name = f"consult-{uuid4().hex}"
    doctor_phone = normalize_au_phone((os.getenv("DOCTOR_PHONE") or "").strip())
    patient_twiml_url = urljoin(base_url_norm, f"twiml/join-consult?room={conference_name}&role=patient")
    doctor_twiml_url = urljoin(base_url_norm, f"twiml/join-consult?room={conference_name}&role=doctor")

    try:
        if doctor_phone:
            doctor_res = http.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json",
                auth=(account_sid, auth_token),
                data={
                    "To": doctor_phone,
                    "From": twilio_number,
                    "Url": doctor_twiml_url,
                    "StatusCallback": status_url,
                    "StatusCallbackMethod": "POST",
                    "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
                },
                timeout=20,
            )
            if doctor_res.status_code >= 400:
                detail = doctor_res.json().get("message") if "application/json" in doctor_res.headers.get("content-type", "") else doctor_res.text
                return jsonify({"ok": False, "error": f"Twilio doctor leg failed: {detail}"}), 502
            doctor_sid = doctor_res.json().get("sid")
        else:
            doctor_sid = None

        patient_res = http.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls.json",
            auth=(account_sid, auth_token),
            data={
                "To": patient_phone,
                "From": twilio_number,
                "Url": patient_twiml_url,
                "StatusCallback": status_url,
                "StatusCallbackMethod": "POST",
                "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
            },
            timeout=20,
        )
        if patient_res.status_code >= 400:
            detail = patient_res.json().get("message") if "application/json" in patient_res.headers.get("content-type", "") else patient_res.text
            return jsonify({"ok": False, "error": f"Twilio patient leg failed: {detail}"}), 502
        patient_sid = patient_res.json().get("sid")
        return jsonify({"ok": True, "room": conference_name, "patientSid": patient_sid, "doctorSid": doctor_sid, "to": patient_phone}), 200
    except requests.RequestException as exc:
        return jsonify({"ok": False, "error": f"Twilio request failed: {exc}"}), 502


@app.post("/twiml/connect-patient")
def twiml_connect_patient():
    stream_url = (os.getenv("STREAM_URL") or "").strip()
    if stream_url:
        xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="{stream_url}" /></Connect></Response>'
    else:
        xml = '<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="alice">Please hold while we connect your consultation.</Say><Pause length="60"/></Response>'
    return make_response(xml, 200, {"Content-Type": "text/xml; charset=utf-8"})


@app.post("/twiml/join-consult")
def twiml_join_consult():
    room = (request.args.get("room") or f"consult-{uuid4().hex}").strip()
    role = (request.args.get("role") or "participant").strip().lower()
    start_on_enter = "true" if role == "doctor" else "false"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Dial><Conference "
        f'startConferenceOnEnter="{start_on_enter}" '
        'endConferenceOnExit="false" '
        'beep="false">'
        f"{room}</Conference></Dial></Response>"
    )
    return make_response(xml, 200, {"Content-Type": "text/xml; charset=utf-8"})


@app.post("/api/call-status")
def call_status():
    monitor.record_system_metric("twilio.call_status.webhook", 1.0)
    return "", 204


def normalize_au_phone(phone: str) -> str:
    cleaned = re.sub(r"[^\d+]", "", phone or "")
    if cleaned.startswith("+") and re.fullmatch(r"\+\d{8,15}", cleaned):
        return cleaned
    digits = re.sub(r"\D", "", cleaned)
    if digits.startswith("61") and len(digits) == 11:
        return f"+{digits}"
    if digits.startswith("0") and len(digits) == 10:
        return f"+61{digits[1:]}"
    return ""

_whisper_model = None
_whisper_init_lock = threading.Lock()
_transcribe_lock = threading.Lock()

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_init_lock:
            if _whisper_model is None:
                _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model

CLINICAL_SYSTEM_PROMPT = (
    "You are an Australian clinical education assistant for qualified medical doctors.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\nFollow-up & Safety Netting\nRed Flags\nReferences\n\n"
    "STYLE:\n"
    "Plain text only. Registrar-level depth. Australian practice framing.\n"
    "If the user pastes mixed notes/results, organise them cleanly under the correct headings.\n"
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

CONSULT_NOTE_SYSTEM_PROMPT = (
    "You are an Australian clinician assistant.\n\n"
    "Task: Convert the provided raw dictation/pasted data into a high-quality clinical note.\n"
    "If content is messy or partial, infer structure but do not invent facts.\n"
    "Use Australian spelling.\n"
    "Optimise for clinical utility: concise, specific, and action-oriented.\n"
    "Carry forward key positives/negatives and unresolved risks when present in input.\n"
    "When details are missing, write 'Not documented' instead of guessing.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\nFollow-up & Safety Netting\nRed Flags\nReferences\n"
)

HANDOVER_SYSTEM_PROMPT = (
    "You are an Australian emergency medicine handover assistant.\n\n"
    "Task: Produce a crisp handover/presentation from the provided raw dictation/pasted data.\n"
    "Primary default is ED handover, BUT if the content clearly matches another context "
    "(e.g., ward round, ICU, theatre, psych, GP), adapt the handover style accordingly.\n"
    "Do not invent facts.\n"
    "Make it usable for verbal handover.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\nFollow-up & Safety Netting\nRed Flags\nReferences\n"
)

CONSULT_TYPE_INSTRUCTIONS = {
    "weight loss initial consult": "Focus on baseline obesity/metabolic history, contraindications, goals, and initial management plan.",
    "weight loss follow-up": "Focus on response to treatment, adverse effects, adherence, dose changes, and next-step plan.",
    "medicinal cannabis / cbd / thc consult": "Focus on indication, prior therapies, contraindications, product rationale, risk counselling, and monitoring plan.",
    "chronic pain consult": "Focus on pain mechanism, function impact, multimodal strategy, opioid risk mitigation, and follow-up.",
    "mental health review": "Focus on mental state, risk assessment, functioning, diagnosis refinement, and safety plan.",
    "men’s health consult": "Focus on men's health concerns, sexual/reproductive history, cardiovascular/metabolic risk, and shared plan.",
    "dva allied health referral": "Use DVA referral framing including accepted conditions, referral rationale, renewal checks, and audit readiness.",
    "gp letter": "Format as a GP letter with reason for correspondence, summary, assessment, actions, and requested follow-up.",
    "patient instructions": "Write clear patient-facing instructions: diagnosis summary, medicines, self-care, warning signs, and when to seek help.",
    "medical certificate / work capacity note": "Focus on work capacity, restrictions, likely duration, review timing, and legal/clinical clarity.",
    "pathology review": "Focus on abnormal/normal result interpretation, clinical significance, differential, and actionable plan.",
    "medication follow-up": "Focus on efficacy, side effects, adherence, interactions, and medicine optimisation plan.",
    "emergency department note": "Use standard Australian ED admission note structure with Presenting Complaint, History of Presenting Complaint, Medical History, Social History, Medications, Allergies, On Examination, Investigations, and Plan.",
    "general consultation note": "Use a comprehensive general consultation note structure suitable for routine primary care.",
}


def build_consult_prompt_context(consult_type: str) -> str:
    normalized = (consult_type or "").strip().lower()
    chosen_type = normalized or "general consultation note"
    guidance = CONSULT_TYPE_INSTRUCTIONS.get(chosen_type, CONSULT_TYPE_INSTRUCTIONS["general consultation note"])
    if chosen_type == "emergency department note":
        return (
            f"Consult type selected: {chosen_type}.\n"
            f"Structure emphasis: {guidance}\n"
            "Use this exact heading order for this note type:\n"
            "Presenting Complaint\n"
            "History of Presenting Complaint\n"
            "Medical History\n"
            "Social History\n"
            "Medications\n"
            "Allergies\n"
            "On Examination\n"
            "Investigations\n"
            "Plan\n"
            "For any missing section details, write 'Not documented'."
        )

    return (
        f"Consult type selected: {chosen_type}.\n"
        f"Structure emphasis: {guidance}\n"
        "Ensure headings and ordering are appropriate to this consult type.\n"
        "Prioritise concise, clinically actionable output and do not invent missing facts."
    )

def call_deepseek(system_prompt: str, user_content: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

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

    resp = http.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=70)
    resp.raise_for_status()
    out = resp.json()
    answer = (((out.get("choices") or [{}])[0]).get("message", {}) or {}).get("content", "").strip()
    return answer or "No response."

@app.get("/", endpoint="index")
def index():
    if session.get("authenticated") is True:
        resp = make_response(render_template("consultation-notes.html"))
        return resp
    return redirect(url_for("login"))


@app.get("/consultation-notes")
@require_auth
def consultation_notes():
    return render_template("consultation-notes.html")


@app.get("/dashboard")
@require_auth
def dashboard():
    return render_template("dashboard.html")


@app.get("/history")
@require_auth
def history():
    return render_template("history.html")


@app.get("/login")
def login():
    if session.get("authenticated") is True:
        return redirect(url_for("consultation_notes"))
    return render_template("login.html")


@app.get("/api/session")
def api_session():
    return jsonify({"ok": True})

@app.get("/api/me")
def api_me():
    is_authenticated = session.get("authenticated") == True
    return jsonify({
        "logged_in": is_authenticated,
        "plan": "pro" if is_authenticated else "guest",
        "used": 0,
        "limit": 1000000 if is_authenticated else 10,
        "remaining": 1000000 if is_authenticated else 10,
    })


@app.get("/api/perf/summary")
@require_auth
def perf_summary():
    stats = monitor.get_stats() or {}
    return jsonify({
        "uptime_seconds": round(stats.get("uptime_seconds", 0), 2),
        "requests": stats.get("requests", 0),
        "avg_duration_ms": round(stats.get("avg_duration_ms", 0), 2),
        "p95_duration_ms": round(stats.get("p95_duration_ms", 0), 2),
    })


@app.get("/api/perf/health")
@require_auth
def perf_health():
    return jsonify({
        "status": "ok",
        "tracked_endpoints": len(monitor.metrics),
    })


@app.get("/api/perf/stats")
@require_auth
def perf_stats():
    return jsonify({"endpoints": monitor.get_all_stats()})

@app.post("/authenticate")
def authenticate():
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip()
    if code == AUTH_CODE:
        session["authenticated"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Invalid code"}), 401

@app.post("/auth/logout")
def auth_logout():
    session.clear()
    return jsonify({"ok": True})

@app.post("/api/generate")
@require_auth
def generate():
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    mode = (data.get("mode") or "clinical").strip().lower()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        if mode.startswith("dva"):
            referral_intent = "D0904 new" if mode == "dva_new" else "D0904 renewal" if mode == "dva_renew" else "D0904 (unspecified)"
            user_content = (
                f"Referral intent: {referral_intent}\n\n"
                f"DETAILS:\n{query}\n\n"
                "Follow DVA_META format then clinical headings."
            )
            answer = call_deepseek(DVA_SYSTEM_PROMPT, user_content)
        else:
            user_content = f"Clinical question:\n{query}\n\nIf pasted data is included, sort it into the correct headings."
            answer = call_deepseek(CLINICAL_SYSTEM_PROMPT, user_content)

        return jsonify({"answer": answer})

    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "AI request failed"}), 502


@app.post("/ask")
@require_auth
def ask_legacy():
    """Backward-compatible endpoint used by consultation-notes.html."""
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    context = (data.get("context") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        user_content = f"Clinical question:\n{question}"
        if context:
            user_content += f"\n\nRecent context:\n{context}"
        user_content += "\n\nIf pasted data is included, sort it into the correct headings."
        answer = call_deepseek(CLINICAL_SYSTEM_PROMPT, user_content)
        save_history("question", answer)
        return jsonify({"answer": answer})
    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "AI request failed"}), 502

@app.post("/api/consult")
@require_auth
def consult():
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    mode = (data.get("mode") or "consult_note").strip().lower()
    consult_type = (data.get("consult_type") or "general consultation note").strip()

    if not text:
        return jsonify({"error": "Empty input"}), 400

    try:
        if mode == "handover":
            user_content = (
                "Create a handover/presentation from the following raw dictation/pasted data. "
                "If the context is not ED, adapt appropriately.\n\n"
                f"{text}"
            )
            answer = call_deepseek(HANDOVER_SYSTEM_PROMPT, user_content)
        else:
            user_content = (
                "Create a structured clinical note from the following raw dictation/pasted data. "
                "Do not invent facts; organise clearly.\n\n"
                f"{build_consult_prompt_context(consult_type)}\n\n"
                f"{text}"
            )
            answer = call_deepseek(CONSULT_NOTE_SYSTEM_PROMPT, user_content)

        return jsonify({"answer": answer})

    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "AI request failed"}), 502


@app.post("/convert-notes")
@require_auth
def convert_notes_legacy():
    """Backward-compatible endpoint used by consultation-notes.html."""
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("clinical_data") or "").strip()
    note_type = (data.get("note_type") or "consultation_note").strip().lower()
    consult_type = (data.get("consult_type") or "general consultation note").strip()
    if not text:
        return jsonify({"error": "Empty input"}), 400

    try:
        mode = "handover" if note_type == "handover" else "consult_note"
        if mode == "handover":
            user_content = (
                "Create a handover/presentation from the following raw dictation/pasted data. "
                "If the context is not ED, adapt appropriately.\n\n"
                f"{text}"
            )
            answer = call_deepseek(HANDOVER_SYSTEM_PROMPT, user_content)
        else:
            user_content = (
                "Create a structured clinical note from the following raw dictation/pasted data. "
                "Do not invent facts; organise clearly.\n\n"
                f"{build_consult_prompt_context(consult_type)}\n\n"
                f"{text}"
            )
            answer = call_deepseek(CONSULT_NOTE_SYSTEM_PROMPT, user_content)
        save_history("note", answer)
        return jsonify({"clinical_notes": answer})
    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "AI request failed"}), 502


@app.post("/auth/google")
def auth_google_not_configured():
    # Explicit response avoids silent 404s in the legacy UI.
    return jsonify({"ok": False, "error": "Google auth is not configured in this build"}), 501


@app.post("/api/stripe/create-checkout-session")
@require_auth
def stripe_checkout_not_configured():
    # Explicit response avoids silent 404s in the legacy UI.
    return jsonify({"error": "Stripe checkout is not configured in this build"}), 501

@app.post("/api/transcribe")
@require_auth
def transcribe():
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "Missing audio"}), 400

    with _transcribe_lock:
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".webm")
            os.close(fd)
            f.save(tmp_path)

            wav_path = tmp_path + ".wav"
            cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            model = get_whisper_model()
            segments, _info = model.transcribe(wav_path, beam_size=5, vad_filter=True)

            text = " ".join((seg.text or "").strip() for seg in segments).strip()
            return jsonify({"text": text})

        except Exception as e:
            print("TRANSCRIBE ERROR:", repr(e))
            return jsonify({"error": "Transcription failed"}), 500
        finally:
            for p in [tmp_path, (tmp_path + ".wav") if tmp_path else None]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass


@app.get("/api/history/list")
@require_auth
def api_history_list():
    return jsonify({"items": load_history()})




@app.post("/api/history/delete")
@require_auth
def api_history_delete():
    data = request.get_json(silent=True) or {}
    entry_id = data.get("id")
    if not isinstance(entry_id, int):
        return jsonify({"error": "Invalid id"}), 400
    if not delete_history_entry(entry_id):
        return jsonify({"error": "Not found"}), 404
    return jsonify({"ok": True})


@app.post("/api/history/clear")
@require_auth
def api_history_clear():
    deleted = clear_history_entries()
    return jsonify({"ok": True, "deleted": deleted})

@app.post("/api/history/save")
@require_auth
def api_history_save():
    data = request.get_json(silent=True) or {}
    item_type = (data.get("type") or "").strip().lower()
    content = (data.get("content") or "").strip()
    if item_type not in {"note", "question"}:
        return jsonify({"error": "Invalid type"}), 400
    if not content:
        return jsonify({"error": "Empty content"}), 400
    save_history(item_type, content)
    return jsonify({"ok": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
