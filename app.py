import os
import re
import time
import tempfile
import threading
import subprocess
import sqlite3
import json
import base64
import hashlib
import hmac
import html
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
from functools import wraps
from urllib.parse import urlencode, urljoin, urlparse

import requests
import websocket
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
from flask_sock import Sock

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
EXTENSION_SYNC_TOKEN = (os.getenv("EXTENSION_SYNC_TOKEN") or "").strip()
MAX_AUDIO_UPLOAD_BYTES = int(os.getenv("MAX_AUDIO_UPLOAD_MB") or "25") * 1024 * 1024
FFMPEG_TIMEOUT_SECONDS = int(os.getenv("FFMPEG_TIMEOUT_SECONDS") or "60")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "dev-insecure-change-me"
app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_UPLOAD_BYTES
sock = Sock(app)

http = requests.Session()
transcript_clients = set()
transcript_clients_lock = threading.Lock()
active_transcript_streams = 0
active_transcript_lock = threading.Lock()


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


def env_flag(name: str, default: bool = True) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def twilio_signature_url() -> str:
    configured_base = (os.getenv("BASE_URL") or os.getenv("APP_BASE_URL") or "").strip()
    if configured_base:
        base_url = configured_base if configured_base.endswith("/") else f"{configured_base}/"
        signed_url = urljoin(base_url, request.path.lstrip("/"))
        if request.query_string:
            signed_url = f"{signed_url}?{request.query_string.decode('utf-8')}"
        return signed_url
    return request.url


def valid_twilio_signature(auth_token: str) -> bool:
    signature = (request.headers.get("X-Twilio-Signature") or "").strip()
    if not signature:
        return False
    signed_data = twilio_signature_url()
    for key, value in sorted(request.form.items(multi=True)):
        signed_data += f"{key}{value}"
    digest = hmac.new(auth_token.encode("utf-8"), signed_data.encode("utf-8"), hashlib.sha1).digest()
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, signature)


def twilio_validation_error_response():
    auth_token = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    if not auth_token or not env_flag("TWILIO_VALIDATE_SIGNATURE", True):
        return None
    if valid_twilio_signature(auth_token):
        return None
    app.logger.warning("Rejected unsigned or invalid Twilio webhook request for %s", request.path)
    return make_response("Forbidden", 403)


def twilio_stream_secret_misconfigured_response():
    stream_secret = (os.getenv("TWILIO_STREAM_SECRET") or "").strip()
    auth_token = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    if stream_secret and (not auth_token or not env_flag("TWILIO_VALIDATE_SIGNATURE", True)):
        app.logger.error("TWILIO_STREAM_SECRET requires TWILIO_AUTH_TOKEN and signature validation")
        return make_response("Twilio stream secret requires signed Twilio webhooks", 503)
    return None


def db_conn():
    timeout = float(os.getenv("SQLITE_TIMEOUT_SECONDS") or "30")
    conn = sqlite3.connect(DB_PATH, timeout=timeout, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return conn


def init_history_db():
    with db_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS medirecords_sync_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'extension',
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


def extension_sync_authorized(payload=None) -> bool:
    if not EXTENSION_SYNC_TOKEN:
        return False
    header = request.headers.get("Authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        token = request.headers.get("X-VividMedi-Sync-Token", "").strip()
    if not token and isinstance(payload, dict):
        token = str(payload.get("syncToken") or payload.get("token") or "").strip()
    return token == EXTENSION_SYNC_TOKEN


def save_medirecords_sync(payload: dict, source: str = "extension"):
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO medirecords_sync_entries (user_key, payload, source, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ("extension", json.dumps(payload), source, datetime.utcnow().isoformat()),
        )
        conn.commit()


def latest_medirecords_sync():
    with db_conn() as conn:
        row = conn.execute(
            """
            SELECT id, payload, source, created_at
            FROM medirecords_sync_entries
            WHERE user_key = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            ("extension",),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "payload": json.loads(row["payload"]),
        "source": row["source"],
        "created_at": row["created_at"],
    }


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
    twilio_number_raw = (os.getenv("TWILIO_NUMBER") or "").strip()
    twilio_number = normalize_twilio_from_phone(twilio_number_raw)
    base_url = twilio_base_url()

    missing = [
        name for name, value in (
            ("TWILIO_ACCOUNT_SID", account_sid),
            ("TWILIO_AUTH_TOKEN", auth_token),
            ("TWILIO_NUMBER", twilio_number_raw),
        )
        if not value
    ]
    if missing:
        return jsonify({
            "ok": False,
            "error": f"Calling is not configured. Missing: {', '.join(missing)}."
        }), 503
    if not twilio_number:
        return jsonify({"ok": False, "error": "TWILIO_NUMBER must be a valid E.164 phone number, e.g. +614XXXXXXXX."}), 503

    payload = request.get_json(silent=True) or {}
    patient_phone_raw = str(payload.get("patientPhone") or "").strip()
    patient_phone = normalize_au_phone(patient_phone_raw)
    if not patient_phone:
        return jsonify({"ok": False, "error": "Invalid patientPhone. Use E.164 (e.g. +614XXXXXXXX) or AU mobile format."}), 400

    if not is_public_twilio_base_url(base_url):
        return jsonify({
            "ok": False,
            "error": "Calling needs a public BASE_URL or APP_BASE_URL, e.g. https://www.vividmedi.com. Local preview URLs cannot receive Twilio webhooks."
        }), 503

    base_url_norm = base_url if base_url.endswith("/") else f"{base_url}/"
    status_url = urljoin(base_url_norm, "api/call-status")
    conference_name = f"consult-{uuid4().hex}"
    doctor_phone = normalize_au_phone((os.getenv("DOCTOR_PHONE") or "").strip())
    if not doctor_phone and (os.getenv("ALLOW_PATIENT_ONLY_CALLS") or "").strip().lower() not in {"1", "true", "yes"}:
        return jsonify({
            "ok": False,
            "error": "Doctor phone is not configured. Set DOCTOR_PHONE so Twilio can bridge your phone with the patient."
        }), 503

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
                    "Method": "POST",
                    "StatusCallback": status_url,
                    "StatusCallbackMethod": "POST",
                    "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
                },
                timeout=20,
            )
            if doctor_res.status_code >= 400:
                detail = twilio_response_error(doctor_res)
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
                "Method": "POST",
                "StatusCallback": status_url,
                "StatusCallbackMethod": "POST",
                "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
            },
            timeout=20,
        )
        if patient_res.status_code >= 400:
            detail = twilio_response_error(patient_res)
            return jsonify({"ok": False, "error": f"Twilio patient leg failed: {detail}"}), 502
        patient_sid = patient_res.json().get("sid")
        return jsonify({"ok": True, "room": conference_name, "sid": patient_sid, "patientSid": patient_sid, "doctorSid": doctor_sid, "to": patient_phone}), 200
    except requests.RequestException as exc:
        app.logger.exception("Twilio call request failed")
        return jsonify({"ok": False, "error": f"Twilio request failed: {exc}"}), 502


@app.route("/twiml/connect-patient", methods=["GET", "POST"])
def twiml_connect_patient():
    validation_error = twilio_validation_error_response()
    if validation_error:
        return validation_error
    stream_url = (os.getenv("STREAM_URL") or "").strip()
    if stream_url:
        xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="{stream_url}" /></Connect></Response>'
    else:
        xml = '<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="alice">Please hold while we connect your consultation.</Say><Pause length="60"/></Response>'
    return make_response(xml, 200, {"Content-Type": "text/xml; charset=utf-8"})


@app.route("/twiml/join-consult", methods=["GET", "POST"])
def twiml_join_consult():
    validation_error = twilio_validation_error_response()
    if validation_error:
        return validation_error
    secret_error = twilio_stream_secret_misconfigured_response()
    if secret_error:
        return secret_error
    room = (request.args.get("room") or f"consult-{uuid4().hex}").strip()
    role = (request.args.get("role") or "participant").strip().lower()
    start_on_enter = "true" if role == "doctor" else "false"
    stream_url = twilio_media_stream_url(room, role)
    stream_status_url = twilio_stream_status_url(room, role)
    stream_track = twilio_stream_track(role)
    stream_xml = (
        '<Start><Stream '
        f'name="{html.escape(twilio_stream_name(room, role), quote=True)}" '
        f'url="{html.escape(stream_url, quote=True)}" '
        f'track="{html.escape(stream_track, quote=True)}" '
        f'statusCallback="{html.escape(stream_status_url, quote=True)}" '
        'statusCallbackMethod="POST">'
        f'<Parameter name="room" value="{html.escape(room, quote=True)}" />'
        f'<Parameter name="role" value="{html.escape(role, quote=True)}" />'
        f"{twilio_stream_secret_parameter()}"
        '</Stream></Start>'
        if stream_url and should_start_media_stream(role) else ""
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response>{stream_xml}<Dial><Conference "
        f'startConferenceOnEnter="{start_on_enter}" '
        'endConferenceOnExit="false" '
        'beep="false">'
        f"{html.escape(room)}</Conference></Dial></Response>"
    )
    return make_response(xml, 200, {"Content-Type": "text/xml; charset=utf-8"})


@app.post("/api/stream-status")
def stream_status():
    validation_error = twilio_validation_error_response()
    if validation_error:
        return validation_error
    event = (request.form.get("StreamEvent") or "stream-status").strip()
    error = (request.form.get("StreamError") or "").strip()
    role = (request.args.get("role") or "call").strip().lower()
    stream_name = (request.form.get("StreamName") or "").strip()
    app.logger.info("Twilio media stream status: event=%s role=%s stream=%s error=%s", event, role, stream_name, error)
    if event == "stream-started":
        broadcast_transcript_status("stream", f"Twilio audio stream started for {role}.")
    elif event == "stream-error":
        broadcast_transcript_status("error", f"Twilio audio stream error: {error or 'unknown error'}")
    elif event == "stream-stopped":
        broadcast_transcript_status("stopped", f"Twilio audio stream stopped for {role}.")
    return "", 204


@app.get("/api/transcription-health")
@require_auth
def transcription_health():
    with transcript_clients_lock:
        client_count = len(transcript_clients)
    with active_transcript_lock:
        active_streams = active_transcript_streams
    return jsonify({
        "deepgramConfigured": bool((os.getenv("DEEPGRAM_API_KEY") or "").strip()),
        "streamSecretConfigured": bool((os.getenv("TWILIO_STREAM_SECRET") or "").strip()),
        "appBaseUrl": twilio_base_url(),
        "streamUrl": twilio_media_stream_url("diagnostic", "doctor"),
        "streamLeg": (os.getenv("TWILIO_STREAM_LEG") or "doctor").strip().lower(),
        "streamTrack": twilio_stream_track("doctor"),
        "frontendTranscriptClients": client_count,
        "activeTranscriptStreams": active_streams,
    })


@app.post("/api/call-status")
def call_status():
    validation_error = twilio_validation_error_response()
    if validation_error:
        return validation_error
    monitor.record_system_metric("twilio.call_status.webhook", 1.0)
    return "", 204


@sock.route("/frontend-transcript")
def frontend_transcript(ws):
    if session.get("authenticated") is not True:
        try:
            ws.close()
        except Exception:
            pass
        return
    with transcript_clients_lock:
        transcript_clients.add(ws)
    send_transcript_message(ws, {"type": "status", "status": "connected"})
    send_transcript_message(ws, {"type": "status", "status": "active" if active_transcript_streams else "stopped"})
    try:
        while True:
            message = ws.receive()
            if message is None:
                break
    finally:
        with transcript_clients_lock:
            transcript_clients.discard(ws)


@sock.route("/twilio-stream")
def twilio_stream(ws):
    role = (request.args.get("role") or "participant").strip().lower()
    room = (request.args.get("room") or "").strip()
    deepgram_api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
    deepgram_streams = {}
    deepgram_failed_tracks = set()
    deepgram_reader_stop = threading.Event()
    stream_started = False
    transcription_marked_active = False
    audio_status_sent = False

    if not deepgram_api_key:
        broadcast_transcript_status("error", "DEEPGRAM_API_KEY is not configured.")

    def get_deepgram_stream(track: str):
        track_key = (track or "inbound").strip().lower().replace("_track", "")
        if track_key in deepgram_streams:
            return deepgram_streams[track_key]
        if track_key in deepgram_failed_tracks or not deepgram_api_key:
            return None
        speaker_label = twilio_track_speaker_label(role, track_key)
        try:
            dg_ws = websocket.create_connection(
                deepgram_listen_url(),
                header=[f"Authorization: Token {deepgram_api_key}"],
                timeout=10,
            )
            dg_ws.settimeout(None)
        except Exception as exc:
            deepgram_failed_tracks.add(track_key)
            app.logger.exception("Deepgram stream connection failed for %s track", track_key)
            broadcast_transcript_status("error", f"Deepgram transcription connection failed: {exc}")
            return None

        def read_deepgram():
            while not deepgram_reader_stop.is_set():
                try:
                    raw = dg_ws.recv()
                except Exception:
                    break
                handle_deepgram_message(raw, speaker_label)

        threading.Thread(target=read_deepgram, daemon=True).start()
        deepgram_streams[track_key] = dg_ws
        app.logger.info("Deepgram stream connected: role=%s track=%s speaker=%s", role, track_key, speaker_label)
        return dg_ws

    try:
        while True:
            raw = ws.receive()
            if raw is None:
                break
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue
            event = message.get("event")
            if event == "start" and not stream_started:
                custom_parameters = message.get("start", {}).get("customParameters", {}) or {}
                expected_secret = (os.getenv("TWILIO_STREAM_SECRET") or "").strip()
                received_secret = (custom_parameters.get("streamSecret") or "").strip()
                if expected_secret and received_secret != expected_secret:
                    app.logger.warning("Rejected Twilio media stream with invalid stream secret")
                    break
                role = (custom_parameters.get("role") or role).strip().lower()
                room = (custom_parameters.get("room") or room).strip()
                stream_started = True
                broadcast_transcript_status("stream", "Call audio stream connected. Waiting for speech...")
                if deepgram_api_key:
                    transcription_marked_active = True
                    increment_transcript_streams()
                else:
                    broadcast_transcript_status("error", "Call audio stream connected, but Deepgram transcription is not available.")
                app.logger.info("Twilio media stream started: role=%s room=%s", role, room)
            elif event == "media" and stream_started:
                media = message.get("media", {}) or {}
                payload = media.get("payload")
                track = media.get("track") or "inbound"
                if payload:
                    try:
                        if not audio_status_sent:
                            audio_status_sent = True
                            broadcast_transcript_status("audio", "Receiving call audio. Waiting for transcription...")
                        dg_ws = get_deepgram_stream(track)
                        if dg_ws:
                            dg_ws.send_binary(base64.b64decode(payload))
                    except Exception as exc:
                        app.logger.warning("Failed to forward Twilio audio to Deepgram: %s", exc)
            elif event == "stop":
                break
    except Exception as exc:
        app.logger.exception("Twilio transcription stream failed")
        broadcast_transcript_status("error", f"Call transcription failed: {exc}")
    finally:
        deepgram_reader_stop.set()
        if transcription_marked_active:
            decrement_transcript_streams()
        for dg_ws in list(deepgram_streams.values()):
            try:
                dg_ws.close()
            except Exception:
                pass


def send_transcript_message(ws, payload) -> bool:
    try:
        ws.send(json.dumps(payload))
        return True
    except Exception:
        return False


def broadcast_transcript(payload):
    with transcript_clients_lock:
        clients = list(transcript_clients)
    stale = []
    for client in clients:
        if not send_transcript_message(client, payload):
            stale.append(client)
    if stale:
        with transcript_clients_lock:
            for client in stale:
                transcript_clients.discard(client)


def broadcast_transcript_status(status: str, message: str = ""):
    payload = {"type": "status", "status": status}
    if message:
        payload["message"] = message
    broadcast_transcript(payload)


def increment_transcript_streams():
    global active_transcript_streams
    with active_transcript_lock:
        active_transcript_streams += 1
    broadcast_transcript_status("active")


def decrement_transcript_streams():
    global active_transcript_streams
    with active_transcript_lock:
        active_transcript_streams = max(0, active_transcript_streams - 1)
        still_active = active_transcript_streams > 0
    broadcast_transcript_status("active" if still_active else "stopped")


def deepgram_listen_url() -> str:
    params = {
        "model": os.getenv("DEEPGRAM_MODEL", "nova-2"),
        "language": os.getenv("DEEPGRAM_LANGUAGE", "en-AU"),
        "encoding": "mulaw",
        "sample_rate": "8000",
        "channels": "1",
        "interim_results": "false",
        "punctuate": "true",
        "smart_format": "true",
    }
    return f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"


def deepgram_prerecorded_url() -> str:
    params = {
        "model": os.getenv("DEEPGRAM_MODEL", "nova-2"),
        "language": os.getenv("DEEPGRAM_LANGUAGE", "en-AU"),
        "punctuate": "true",
        "smart_format": "true",
    }
    return f"https://api.deepgram.com/v1/listen?{urlencode(params)}"


def extract_deepgram_transcript(payload: dict) -> str:
    return (
        payload.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
        .strip()
    )


def transcribe_audio_with_deepgram(audio_bytes: bytes, content_type: str) -> str:
    deepgram_api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
    if not deepgram_api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured")

    response = http.post(
        deepgram_prerecorded_url(),
        headers={
            "Authorization": f"Token {deepgram_api_key}",
            "Content-Type": content_type or "audio/webm",
        },
        data=audio_bytes,
        timeout=60,
    )
    if response.status_code >= 400:
        detail = response.text[:300] if response.text else f"HTTP {response.status_code}"
        raise RuntimeError(f"Deepgram transcription failed: {detail}")
    return extract_deepgram_transcript(response.json())


def handle_deepgram_message(raw, role_label: str):
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return
    transcript = (
        data.get("channel", {})
        .get("alternatives", [{}])[0]
        .get("transcript", "")
        .strip()
    )
    if not transcript:
        return
    if data.get("is_final") is False:
        return
    broadcast_transcript({"type": "transcript", "text": f"{role_label}: {transcript}", "speaker": role_label})


def twilio_base_url() -> str:
    configured = (os.getenv("BASE_URL") or os.getenv("APP_BASE_URL") or "").strip()
    base_url = configured or request.host_url
    if base_url and not re.match(r"^https?://", base_url, flags=re.IGNORECASE):
        base_url = f"https://{base_url}"
    return base_url if base_url.endswith("/") else f"{base_url}/"


def twilio_media_stream_url(room: str, role: str) -> str:
    configured = (os.getenv("STREAM_URL") or "").strip()
    if configured:
        stream_url = configured
    else:
        base_url = twilio_base_url()
        stream_url = urljoin(base_url, "twilio-stream")
        stream_url = re.sub(r"^https:", "wss:", stream_url, flags=re.IGNORECASE)
        stream_url = re.sub(r"^http:", "ws:", stream_url, flags=re.IGNORECASE)
    return stream_url


def should_start_media_stream(role: str) -> bool:
    mode = (os.getenv("TWILIO_STREAM_LEG") or "doctor").strip().lower()
    role = (role or "").strip().lower()
    if mode in {"both", "all", "*"}:
        return role in {"doctor", "patient", "participant"}
    if mode in {"patient", "patient_only"}:
        return role == "patient"
    if mode in {"off", "disabled", "none"}:
        return False
    return role == "doctor"


def twilio_stream_track(role: str) -> str:
    configured = (os.getenv("TWILIO_STREAM_TRACK") or "").strip().lower()
    allowed = {"inbound_track", "outbound_track", "both_tracks"}
    if configured in allowed:
        return configured
    return "both_tracks"


def twilio_stream_name(room: str, role: str) -> str:
    safe_room = re.sub(r"[^a-zA-Z0-9_-]+", "-", room or "consult").strip("-")[:64]
    safe_role = re.sub(r"[^a-zA-Z0-9_-]+", "-", role or "call").strip("-")[:24]
    return f"{safe_room}-{safe_role}"


def twilio_stream_status_url(room: str, role: str) -> str:
    base_url = twilio_base_url()
    status_url = urljoin(base_url, "api/stream-status")
    params = urlencode({"room": room or "", "role": role or ""})
    return f"{status_url}?{params}"


def twilio_track_speaker_label(role: str, track: str) -> str:
    role = (role or "").strip().lower()
    track = (track or "inbound").strip().lower().replace("_track", "")
    if role == "doctor":
        if track == "inbound":
            return "Clinician"
        if track == "outbound":
            return "Patient"
    if role == "patient":
        if track == "inbound":
            return "Patient"
        if track == "outbound":
            return "Clinician"
    return "Call"


def twilio_stream_secret_parameter() -> str:
    stream_secret = (os.getenv("TWILIO_STREAM_SECRET") or "").strip()
    if not stream_secret:
        return ""
    return f'<Parameter name="streamSecret" value="{html.escape(stream_secret, quote=True)}" />'


def is_public_twilio_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"}:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False
    if host.endswith(".local"):
        return False
    return bool(host)


def twilio_response_error(response) -> str:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = response.json()
            message = body.get("message") or body.get("detail") or response.text
            code = body.get("code")
            return f"{message} (code {code})" if code else str(message)
        except ValueError:
            pass
    return response.text or f"HTTP {response.status_code}"


def normalize_e164_phone(phone: str) -> str:
    cleaned = re.sub(r"[^\d+]", "", phone or "")
    if cleaned.startswith("+") and re.fullmatch(r"\+\d{8,15}", cleaned):
        return cleaned
    return ""


def normalize_twilio_from_phone(phone: str) -> str:
    return normalize_e164_phone(phone) or normalize_au_phone(phone)


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
    "Where consult-type instructions provide an exact organisation-specific structure, "
    "use that structure instead of the default headings.\n"
    "For DVA/AHPRA-sensitive notes, make the documentation defensible and audit-ready, "
    "but do not claim compliance is guaranteed.\n\n"
    "Do not include References or Red Flags sections unless the clinician explicitly asks for them.\n\n"
    "OUTPUT FORMAT (MANDATORY):\n"
    "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\nFollow-up & Safety Netting\n"
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
    "weight loss initial consult": "Use the organisation's initial weight-management consult style. Focus on ID check, telehealth mode, DVA/card context where documented, obesity/metabolic history, comorbidities, current and prior weight-loss medicines, contraindications, counselling, consent/opportunity for questions, baseline observations, starting plan, follow-up, safety-netting, and current medication line.",
    "weight loss follow-up": "Use the organisation's weight-management review/script-renewal style. Focus on response to treatment, adverse effects, tolerability, adherence, current dose, requested dose change/script renewal, BMI/weight trajectory, comorbidities, escalation rationale, dietitian/lifestyle measures, safety-netting, follow-up, and current medication line.",
    "vapac weight loss application": "Write a formal DVA VAPAC application letter for RPBS funding or continuation of weight-loss pharmacotherapy. Focus on patient identifiers, DVA card/file details, accepted conditions/comorbidities, anthropometrics, prior response, 5% weight-loss status for the most recent approval interval, medication history, requested medication/regimen, clinical justification, monitoring, evidence, and missing critical information.",
    "medicinal cannabis / cbd / thc consult": "Focus on indication, prior therapies, contraindications, product rationale, risk counselling, and monitoring plan.",
    "chronic pain consult": "Focus on pain mechanism, function impact, multimodal strategy, opioid risk mitigation, and follow-up.",
    "mental health review": "Focus on mental state, risk assessment, functioning, diagnosis refinement, and safety plan.",
    "wa mental health discharge summary": "Write a polished, comprehensive WA hospital psychiatry discharge summary using NACS-style psychiatric discharge headings. Synthesize multiple pasted admission notes into a coherent consultant-level narrative covering admission problems, psychiatric history, mental state, risk on admission and discharge, formulation, management, investigations, medications, adverse reactions, discharge advice, follow-up, and clear communication to GP/CMHT/patient or carer.",
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

WEIGHT_LOSS_INITIAL_NOTE_STRUCTURE = (
    "Use this exact practical note style for Weight loss initial consult. Plain text only. "
    "Keep it copy-paste ready for the work platform. Do not use Markdown.\n"
    "Heading/order:\n"
    "Telehealth consult\n"
    "ID Verification\n"
    "Patient profile\n"
    "DVA / entitlement context\n"
    "Past medical history\n"
    "Medications\n"
    "Weight management history\n"
    "Current issue / reason for consult\n"
    "Counselling\n"
    "Opportunity to ask questions\n"
    "Assessment\n"
    "Plan\n"
    "Safety Netting\n"
    "Observation\n"
    "Current Medication\n\n"
    "Content requirements:\n"
    "- Use 'Telehealth consult' unless the input clearly says the consult was in-person or another mode.\n"
    "- Include '3 points of ID confirmed' or the specific ID details documented. If not documented, write 'ID verification: Not documented'.\n"
    "- Include age/sex if documented, DVA card colour/accepted conditions if documented, PMHx, medicines, current GLP-1/GIP or other weight-loss medicine, plateau or reason for switch if documented.\n"
    "- Include counselling for tirzepatide/Mounjaro when relevant: once-weekly subcutaneous injection, GLP-1/GIP mechanism, appetite/satiety, insulin sensitivity/metabolic effect, common GI side effects, storage/refrigeration, pen/needle administration, and variable/manageable side effects.\n"
    "- Include contraindications or cautions only if documented. If key contraindication screening is absent, note 'Contraindication screening: Not documented'.\n"
    "- Include patient questions/consent to proceed if documented; otherwise state 'Opportunity to ask questions: Not documented'.\n"
    "- Include baseline weight/height/BMI under Observation when available.\n"
    "- Include a specific starting plan, usually Mounjaro/tirzepatide 2.5 mg weekly if documented or clinically indicated by input, without inventing if unclear.\n"
    "- Include follow-up timing, usually 4 weeks or sooner if concerns when documented/appropriate.\n"
    "- Include a Current Medication line in prescribing-platform style when a drug/dose is documented. If exact formulation/directions are missing, write 'Exact prescribing line: Not documented'."
)

WEIGHT_LOSS_FOLLOWUP_NOTE_STRUCTURE = (
    "Use this exact practical note style for Weight loss follow-up, including script-renewal/dose-adjustment reviews. Plain text only. "
    "Keep it copy-paste ready for the work platform. Do not use Markdown.\n"
    "Heading/order:\n"
    "Consult Type: Script Renewal - Weight Management (<medicine if documented>)\n"
    "Clinician\n"
    "Mode\n"
    "ID Verification\n"
    "Reason for Consult\n"
    "Medical conditions\n"
    "Anthropometrics\n"
    "Current Treatment\n"
    "Progress Since Last Review\n"
    "Side Effects\n"
    "Assessment\n"
    "Plan\n"
    "Safety Netting\n"
    "Current Medication\n\n"
    "Content requirements:\n"
    "- Use the medicine name from the input, e.g. Mounjaro/tirzepatide/Ozempic/semaglutide. If not documented, write 'medicine not documented'.\n"
    "- Include clinician only if documented; otherwise 'Clinician: Not documented'.\n"
    "- Use 'Mode: Telehealth' unless the input clearly says the consult was in-person or another mode.\n"
    "- Include ID verification using full name, DOB and residential address if documented; otherwise write what was confirmed or 'Not documented'.\n"
    "- Include comorbidities/accepted DVA conditions exactly as provided, including OA, OSA, HTN, DM, AF, chronic pain issues, etc. Do not add diagnoses.\n"
    "- Include current weight, height and BMI when available; calculate BMI only if weight and height are available, mark as approximate.\n"
    "- Include current medication and previous/current dose, response, appetite suppression, weight loss/plateau, lifestyle/dietitian input, tolerability, and negative GI/red-flag symptoms when documented.\n"
    "- Assessment should explicitly justify continuing treatment and dose escalation/renewal only when the input supports it: response, BMI/clinical indication, tolerability, no significant adverse effects, ongoing obesity/metabolic risk.\n"
    "- Plan should include exact dose change/script issue when documented, diet/hydration/lifestyle reinforcement, and review timing.\n"
    "- Safety Netting should include stopping injections and seeking urgent care for persistent vomiting, severe abdominal pain especially RUQ, pancreatitis/gallbladder symptoms, or other concerning adverse effects when relevant.\n"
    "- Include a Current Medication line in prescribing-platform style when a drug/dose is documented. If exact formulation/directions are missing, write 'Exact prescribing line: Not documented'."
)

VAPAC_WEIGHT_LOSS_APPLICATION_STRUCTURE = (
    "Use this exact practical letter style for VAPAC weight-loss pharmacotherapy applications. Plain text only. "
    "Keep it copy-paste ready. Do not use Markdown, tables with pipes, or decorative symbols.\n"
    "Use today's date: {today_date}.\n\n"
    "Heading/order:\n"
    "Apex Rx 447 Upper Edward Street\n"
    "Spring Hill, QLD 4000\n"
    "Ph: 1300273979\n"
    "Fax: 0739168300\n"
    "E: contact@apexrx.com.au\n\n"
    "Department of Veterans' Affairs - Application for Funding of Weight Loss Pharmacotherapy Veterans' Affairs Pharmaceutical Advisory Centre (VAPAC)\n\n"
    "{today_date}\n\n"
    "Dear Sirs/ Madams\n\n"
    "<Patient title/name>\n"
    "<DOB>\n"
    "<DVA card type and DVA/file number>\n\n"
    "Starting Weight for Current Approval Interval:\n"
    "Current Weight:\n"
    "Height:\n"
    "BMI:\n"
    "Accepted Conditions / Comorbidities:\n\n"
    "Clinical Summary (Reason for request):\n\n"
    "Current Medication (Generic/ Brand name/ Dose):\n\n"
    "Medication History:\n"
    "Product Name    Dosage    Frequency\n"
    "<convert any pasted medication history into simple aligned plain text rows>\n\n"
    "Requested Medication:\n"
    "Proposed dose and regimen:\n\n"
    "Intended as a maintenance dose, with ongoing clinical review and dose adjustment if required\n"
    "Continued alongside lifestyle measures, dietitian input, and physical activity\n"
    "Planned duration: Ongoing treatment for 4 months, subject to review.\n\n"
    "Monitoring and review:\n"
    "BMI will be used as the primary objective marker for response, with assessment at regular follow-up intervals. "
    "Adjunctive lifestyle measures, including regular exercise and dietitian reviews, will continue. Regular engagement "
    "with the doctor for reviews also acts as a form of check-in and behaviour activation, where the doctor can also "
    "provide informal psychological and medical support in the form of reassurance. It also gives the opportunity for "
    "the doctor to further explore underlying causes for weight gain where clinically relevant, including undiagnosed "
    "ADHD or other contributors that may only become apparent after longitudinal assessment.\n\n"
    "Evidence Supporting Efficacy\n"
    "The following peer-reviewed studies provide evidence supporting the efficacy of once-weekly GLP-1/GIP receptor agonist therapy in adults with overweight or obesity: "
    "'Tirzepatide Once Weekly for the Treatment of Obesity' - New England Journal of Medicine (2022). "
    "'Once-Weekly Semaglutide in Adults with Overweight or Obesity' - New England Journal of Medicine (2021).\n\n"
    "Summary\n"
    "In participants with overweight or obesity, both tirzepatide and semaglutide, administered once weekly alongside lifestyle interventions, were associated with sustained, clinically significant reductions in body weight and improvements in cardiometabolic markers compared to placebo.\n\n"
    "We request a 4 month prescription of medication.\n\n"
    "Additional Notes\n\n"
    "Conclusion\n"
    "This application is made under RPBS arrangements for consideration by the Department of Veterans' Affairs (VAPAC). "
    "The requested treatment is clinically appropriate for this patient's presentation and comorbidity profile, with supporting evidence for efficacy and safety.\n\n"
    "Dr Michael Addis\n"
    "665437AX\n"
    "contact@apex.au\n\n"
    "Critical information missing / issues to address:\n"
    "- <list missing or contradictory items, or write 'No critical missing information identified from the supplied input.'>\n\n"
    "Content requirements:\n"
    "- Preserve supplied patient identifiers, DVA card type, file number, DOB, dates, medications, doses and prior notes accurately.\n"
    "- Calculate BMI if height and current weight are supplied.\n"
    "- For VAPAC continuation, the 5% weight-loss requirement applies to the most recent approved funding interval, generally a 4-month interval / 4 pens. Use the baseline weight at the start of that current approval interval as the denominator, not the original treatment starting weight from earlier months unless that is also the interval baseline.\n"
    "- Medication issue history can identify the approval interval. Treat relevant RPBS weight-loss medication issues (tirzepatide/Mounjaro, semaglutide/Ozempic/Wegovy) as funding pens. Sort issue dates chronologically from oldest to newest and group them into blocks of 4. The baseline date for the most recent completed/attempted interval is the first issue date in the latest 4-pen block.\n"
    "- Example: if the pasted list has 8 weight-loss RPBS issues dated 29/10/2025, 18/11/2025, 23/12/2025, 25/01/2026, 15/02/2026, 15/03/2026, 06/04/2026, 27/04/2026, then the current interval baseline date is 15/02/2026. Do not use a later consult note date or the original 29/10/2025 start date for the 5% denominator.\n"
    "- Use the weight recorded on or closest to that inferred interval baseline date as 'Starting Weight for Current Approval Interval'. If that weight is not supplied, flag it as missing even if older starting weights are available.\n"
    "- If multiple weights/dates are supplied, identify and use the weight at the start of the most recent funded approval interval and the current/authority-attempt weight. Mention older original starting weights only as background.\n"
    "- Explicitly state whether the patient meets or fails the 5% weight-loss continuation threshold for the most recent approval interval when data permits.\n"
    "- If the interval baseline weight is unclear, do not calculate the 5% continuation result from an older original starting weight; instead flag 'current approval interval baseline weight unclear' as critical missing information.\n"
    "- For White Card holders, explicitly link the request to accepted conditions/comorbidities where supplied. If this link is unclear, flag it as critical missing information.\n"
    "- If continuation is requested despite less than 5% weight loss, include a reasoned written-application justification based only on supplied facts: clinical benefits, functional gains, barriers, dose titration, interruptions, tolerability, comorbidity improvement, or risk of harm if ceased.\n"
    "- Do not invent accepted conditions, specialist support, renal/hepatic status, pathology, or medication response. If absent, state it is not documented and flag if important."
)

WA_MENTAL_HEALTH_DISCHARGE_SUMMARY_STRUCTURE = (
    "Use this exact practical document style for WA hospital psychiatry discharge summaries. Plain text only. "
    "Keep it copy-paste ready for a WA Health/NACS style discharge summary. Do not use Markdown. "
    "This document type should be comprehensive and professionally written; do not force it to be short or overly succinct.\n"
    "Heading/order:\n"
    "Event\n"
    "Problems this Admission\n"
    "Clinical Interventions\n"
    "Significant MHx\n"
    "Clinical Synopsis\n"
    "Presenting History\n"
    "Past Psychiatric/Mental Health History\n"
    "Mental State on Admission\n"
    "Risk Assessment on Admission\n"
    "Drug and Alcohol History\n"
    "Family Medical/Mental Health History\n"
    "Social History\n"
    "Developmental and Personal History\n"
    "Admission Physical Assessment (Clinical Findings)\n"
    "Management/Progress (incl. Consultations)\n"
    "Formulation\n"
    "Discharge Mental State\n"
    "Risk Assessment on Discharge\n"
    "Diagnosis\n"
    "Diagnostic Investigations\n"
    "Interpreted Summary\n"
    "Health Profile\n"
    "Adverse Reactions\n"
    "Medications\n"
    "Current Medications\n"
    "Medication Changes / Rationale\n"
    "Discharge Plan\n"
    "Advice to GP\n"
    "Advice to Community Mental Health Team\n"
    "Advice to Residential Aged Care Home\n"
    "Advice to Patient/Guardian/Carer\n"
    "Follow-up / Appointments\n"
    "Safeguarding / Author Notes\n\n"
    "Content requirements:\n"
    "- Expect the input to contain multiple pasted admission notes in messy order. Synthesize them into one coherent discharge summary; do not simply restate each note chronologically.\n"
    "- Build a clinically plausible timeline from dated entries where dates are supplied. If notes conflict, prefer the most recent clearly dated information and flag unresolved contradictions in Safeguarding / Author Notes.\n"
    "- De-duplicate repeated MSE, risk, medication, and collateral material. Keep the most clinically useful final version while preserving important changes over admission.\n"
    "- Be confident in organisation, wording, and summarisation: convert scattered fragments into polished hospital discharge-summary prose. Confidence means clear synthesis, not invented facts.\n"
    "- Use a senior psychiatry registrar/consultant discharge-summary tone: fluent, precise, balanced, and clinically authoritative. The author should sound careful, professional, and across the admission.\n"
    "- Prioritise flow and readability. Where the source material supports it, write developed paragraphs rather than bare fragments, especially in Clinical Synopsis, Presenting History, Management/Progress, Formulation, Discharge Mental State, Risk Assessment on Discharge, and Advice sections.\n"
    "- Do not compress clinically important context merely to be brief. A longer discharge summary is appropriate when it improves continuity of care, risk communication, medication safety, or professional defensibility.\n"
    "- Transform rough pasted notes into finished prose. Avoid phrases that sound like the AI is apologising for missing data. Prefer elegant clinical wording such as 'There was no documentation provided regarding...' only where genuinely needed.\n"
    "- Avoid repetitive 'Not documented' lines when an entire domain is absent. In longer sections, summarise missing domains professionally and only flag critical omissions in Safeguarding / Author Notes.\n"
    "- Use WA hospital discharge-summary tone: factual, handover-oriented, and written for GP, community mental health, patient/carer, and facility readers.\n"
    "- Do not leave a major heading as 'Not documented' if relevant information can reasonably be synthesized from anywhere in the pasted notes. Use 'Not documented' only after considering the whole pasted record.\n"
    "- Preserve the clinician's intended emphasis from the source text. If the author appears to be qualifying risk, uncertainty, capacity, MHA status, diagnosis, substance use, collateral reliability, or family concerns, keep that nuance.\n"
    "- Under Problems this Admission, list principal psychiatric problem first, then comorbidities/complications only when supplied.\n"
    "- Under Clinical Interventions, include inpatient psychiatric care, MHA status, observations, seclusion/restraint, ECT, psychological/OT/social work input, family meetings, discharge planning, and medical reviews only when documented.\n"
    "- Significant MHx should include psychiatric diagnoses, prior admissions, suicide/self-harm history, violence/aggression risk, trauma history, and relevant cognitive/neurodevelopmental history only where documented.\n"
    "- Risk sections should distinguish suicide/self-harm, harm to others, vulnerability/exploitation, absconding, neglect/self-neglect, substance-related risk, and relapse risk where relevant. State static factors, dynamic factors, protective factors, and discharge mitigations when supplied.\n"
    "- Do not write that risk is absent just because it is not mentioned. Use 'Not documented' or 'No evidence documented in the supplied information' as appropriate.\n"
    "- Clinical Synopsis should read as a professional admission-to-discharge narrative: reason for admission, key symptoms/risks, major changes during admission, response to treatment, discharge rationale, and remaining issues.\n"
    "- If the patient died during admission, explicitly adapt the document: state the date/time of death if supplied, describe the final deterioration and comfort-care/palliative approach, use 'Not applicable - patient died during admission' for Discharge Mental State and Risk Assessment on Discharge, and ensure advice/follow-up sections focus on GP/family notification, death certification/cause of death if documented, bereavement/family communication, medication cessation, and administrative handover rather than routine relapse planning.\n"
    "- For complex older-adult psychiatry admissions, integrate medical comorbidity, delirium, dementia/BPSD, falls, pain, infection, nutrition/hydration, capacity/MHA status, goals of care, family meetings, and placement/carer stress into one coherent account.\n"
    "- Management/Progress should integrate medication changes, behavioural observations, engagement, ward course, allied-health/social-work input, family/collateral work, physical-health issues, and discharge planning into a coherent account.\n"
    "- Formulation should be a well-reasoned biopsychosocial formulation tying presentation, vulnerabilities, precipitants, perpetuating factors, protective factors, diagnosis, risk, treatment response, and discharge rationale together.\n"
    "- Diagnosis should be ordered and phrased professionally, separating principal psychiatric/cognitive diagnosis, delirium/medical precipitants, major medical events, injuries, and psychosocial/contextual issues.\n"
    "- Discharge Mental State should be current and specific: appearance/behaviour, rapport, speech, mood/affect, thought form/content, perception, cognition, insight/judgement, and engagement, only from documented material.\n"
    "- Medications should include dose, route, frequency, indication, supply, changes during admission, and monitoring needs when documented. Do not invent reconciliation details.\n"
    "- Advice sections should be practical and directed: GP actions, CMHT follow-up, facility/RACH requirements, patient/carer warning signs, crisis contacts, adherence, monitoring, and when to re-present. Make these sections sound like thoughtful continuity-of-care instructions, not generic filler.\n"
    "- Safeguarding / Author Notes should be brief and should flag missing high-risk information, contradictions, source limitations, or items the author should verify before signing. Do not include defensive boilerplate if no issue is identified; write 'No specific author-safeguarding issues identified from the supplied information.'\n"
    "- Do not provide legal advice, do not claim WA Health compliance is guaranteed, and do not invent dates, diagnoses, MHA status, risk assessments, follow-up appointments, allergies, medication supply, pathology, or collateral."
)


def build_consult_prompt_context(consult_type: str) -> str:
    normalized = (consult_type or "").strip().lower()
    chosen_type = normalized or "general consultation note"
    guidance = CONSULT_TYPE_INSTRUCTIONS.get(chosen_type, CONSULT_TYPE_INSTRUCTIONS["general consultation note"])
    today_date = datetime.now(ZoneInfo("Australia/Perth")).strftime("%d/%m/%Y")
    if chosen_type == "weight loss initial consult":
        return (
            f"Consult type selected: {chosen_type}.\n"
            f"Structure emphasis: {guidance}\n\n"
            f"{WEIGHT_LOSS_INITIAL_NOTE_STRUCTURE}\n\n"
            "Organisation workflow priority:\n"
            "The final note should read like a clinician's work-platform note, not an academic report. "
            "Use short lines and clinically useful headings. Add missing documentation prompts as 'Not documented' "
            "where important for DVA/AHPRA defensibility, without bloating the note."
        )

    if chosen_type == "weight loss follow-up":
        return (
            f"Consult type selected: {chosen_type}.\n"
            f"Structure emphasis: {guidance}\n\n"
            f"{WEIGHT_LOSS_FOLLOWUP_NOTE_STRUCTURE}\n\n"
            "Organisation workflow priority:\n"
            "The final note should read like a script-renewal/weight-management review note suitable for copying into "
            "the clinician's work platform, not an academic report. Use short lines and clinically useful headings. "
            "Add missing documentation prompts as 'Not documented' where important for DVA/AHPRA defensibility, "
            "without bloating the note."
        )

    if chosen_type == "vapac weight loss application":
        return (
            f"Consult type selected: {chosen_type}.\n"
            f"Structure emphasis: {guidance}\n\n"
            f"{VAPAC_WEIGHT_LOSS_APPLICATION_STRUCTURE.format(today_date=today_date)}\n\n"
            "Organisation workflow priority:\n"
            "The final output is a formal application letter to VAPAC, not a routine consult note. "
            "Use the supplied pasted information to populate the letter. Keep it professional, concise and defensible. "
            "For the 5% continuation rule, compare the current weight against the baseline weight for the most recent "
            "approved funding interval, generally the last 4 months / 4 pens, not the original treatment starting "
            "weight from older approvals. If a medication issue list is pasted, infer the current interval start by "
            "sorting RPBS tirzepatide/semaglutide issue dates oldest-to-newest and grouping them into 4-pen blocks; "
            "the first script in the latest 4-pen block anchors the interval baseline date. At the bottom, always "
            "include a Critical information missing / issues to address section."
        )

    if chosen_type == "wa mental health discharge summary":
        return (
            f"Consult type selected: {chosen_type}.\n"
            f"Structure emphasis: {guidance}\n\n"
            f"{WA_MENTAL_HEALTH_DISCHARGE_SUMMARY_STRUCTURE}\n\n"
            "Organisation workflow priority:\n"
            "The final output is a WA hospital psychiatry discharge summary, not a generic mental health review. "
            "Mirror the NACS-style headings, keep the voice clear and clinically familiar for WA psychiatry handover, "
            "and protect the author by preserving uncertainty, collateral/source limits, absent documentation, "
            "and discharge-risk mitigation without overstating certainty. The user may paste many random admission "
            "notes; integrate them into one coherent discharge summary with sensible chronology, de-duplication, "
            "and clinically confident synthesis. This note type should favour a polished, comprehensive final "
            "hospital discharge summary over a short note. If the admission ended in death, adapt all discharge, "
            "risk, advice, and follow-up language accordingly."
        )

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

def consult_completion_budget(consult_type: str) -> int:
    normalized = (consult_type or "").strip().lower()
    if normalized == "wa mental health discharge summary":
        return int(os.getenv("DEEPSEEK_WA_MH_DISCHARGE_MAX_TOKENS") or "6000")
    if normalized == "vapac weight loss application":
        return int(os.getenv("DEEPSEEK_LONG_FORM_MAX_TOKENS") or "3600")
    return int(os.getenv("DEEPSEEK_MAX_TOKENS") or "1800")


def consult_request_timeout(consult_type: str) -> int:
    normalized = (consult_type or "").strip().lower()
    if normalized == "wa mental health discharge summary":
        return int(os.getenv("DEEPSEEK_WA_MH_DISCHARGE_TIMEOUT") or "150")
    return int(os.getenv("DEEPSEEK_TIMEOUT") or "70")


def call_deepseek(system_prompt: str, user_content: str, max_tokens: int | None = None, timeout: int | None = None) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    max_tokens = max_tokens or int(os.getenv("DEEPSEEK_MAX_TOKENS") or "1800")
    timeout = timeout or int(os.getenv("DEEPSEEK_TIMEOUT") or "70")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.25,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = http.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=timeout)
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


@app.get("/patient-list")
@require_auth
def patient_list():
    return render_template("patient-list.html")


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
    if not AUTH_CODE:
        return jsonify({"ok": False, "error": "Authentication is not configured"}), 503
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip()
    if code and hmac.compare_digest(code, AUTH_CODE):
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
            answer = call_deepseek(
                CONSULT_NOTE_SYSTEM_PROMPT,
                user_content,
                max_tokens=consult_completion_budget(consult_type),
                timeout=consult_request_timeout(consult_type),
            )

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
            answer = call_deepseek(
                CONSULT_NOTE_SYSTEM_PROMPT,
                user_content,
                max_tokens=consult_completion_budget(consult_type),
                timeout=consult_request_timeout(consult_type),
            )
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

    if request.content_length and request.content_length > MAX_AUDIO_UPLOAD_BYTES:
        return jsonify({"error": "Audio upload is too large"}), 413

    audio_bytes = f.read(MAX_AUDIO_UPLOAD_BYTES + 1)
    if not audio_bytes:
        return jsonify({"error": "Missing audio"}), 400
    if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        return jsonify({"error": "Audio upload is too large"}), 413

    if (os.getenv("DEEPGRAM_API_KEY") or "").strip():
        try:
            text = transcribe_audio_with_deepgram(audio_bytes, f.mimetype or "audio/webm")
            return jsonify({"text": text})
        except Exception as exc:
            app.logger.warning("Deepgram mic transcription failed, falling back to Whisper: %s", exc)

    with _transcribe_lock:
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".webm")
            os.close(fd)
            with open(tmp_path, "wb") as tmp:
                tmp.write(audio_bytes)

            wav_path = tmp_path + ".wav"
            cmd = ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=FFMPEG_TIMEOUT_SECONDS)

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


@app.post("/api/medirecords-sync")
def api_medirecords_sync_save():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"ok": False, "error": "Expected JSON payload"}), 400

    if not extension_sync_authorized(payload):
        return jsonify({"ok": False, "error": "Unauthorized or EXTENSION_SYNC_TOKEN is not configured"}), 401

    if isinstance(payload, list):
        payload = {"appointments": payload}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Payload must be an object or appointment array"}), 400
    payload.pop("syncToken", None)
    payload.pop("token", None)

    appointments = payload.get("appointments")
    if appointments is None and "patients" not in payload:
        appointments = payload.get("items") or payload.get("data") or payload.get("results")
        if appointments is not None:
            payload["appointments"] = appointments

    if appointments is not None and not isinstance(appointments, list):
        return jsonify({"ok": False, "error": "appointments must be an array"}), 400

    source = str(payload.get("source") or "extension")[:80]
    save_medirecords_sync(payload, source=source)
    return jsonify({
        "ok": True,
        "appointments": len(appointments or []),
        "patients": len(payload.get("patients") or []),
    })


@app.get("/api/medirecords-sync/status")
def api_medirecords_sync_status():
    return jsonify({"ok": True, "tokenConfigured": bool(EXTENSION_SYNC_TOKEN)})


@app.post("/api/medirecords-sync/status")
def api_medirecords_sync_status_check():
    payload = request.get_json(silent=True) or {}
    return jsonify({
        "ok": True,
        "tokenConfigured": bool(EXTENSION_SYNC_TOKEN),
        "tokenAccepted": extension_sync_authorized(payload),
    })


@app.get("/api/medirecords-sync/latest")
@require_auth
def api_medirecords_sync_latest():
    latest = latest_medirecords_sync()
    if latest is None:
        return jsonify({"ok": False, "error": "No MediRecords sync payload found"}), 404
    return jsonify({"ok": True, **latest})




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
