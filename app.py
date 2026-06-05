import os
import re
import time
import tempfile
import threading
import subprocess
import sqlite3
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
from functools import wraps
from urllib.parse import urljoin, urlparse

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
EXTENSION_SYNC_TOKEN = (os.getenv("EXTENSION_SYNC_TOKEN") or "").strip()

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
    stream_url = (os.getenv("STREAM_URL") or "").strip()
    if stream_url:
        xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="{stream_url}" /></Connect></Response>'
    else:
        xml = '<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="alice">Please hold while we connect your consultation.</Say><Pause length="60"/></Response>'
    return make_response(xml, 200, {"Content-Type": "text/xml; charset=utf-8"})


@app.route("/twiml/join-consult", methods=["GET", "POST"])
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


def twilio_base_url() -> str:
    configured = (os.getenv("BASE_URL") or os.getenv("APP_BASE_URL") or "").strip()
    base_url = configured or request.host_url
    if base_url and not re.match(r"^https?://", base_url, flags=re.IGNORECASE):
        base_url = f"https://{base_url}"
    return base_url if base_url.endswith("/") else f"{base_url}/"


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
