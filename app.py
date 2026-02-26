# app.py — VividMedi (Flask) — FULL PASTEABLE VERSION (Stripe SUBSCRIPTION via PRICE ID)
# ✅ Google Sign-In (server verified)
# ✅ Returns Bearer token to frontend (index.html expects data.token)
# ✅ Stripe subscription checkout: $30/month, NO trial, uses PRICE id (price_...)
# ✅ Webhook upgrades user via stripe_customer_id (reliable)
# ✅ Quota tracking (guest/free/pro)
# ✅ DeepSeek generate + consult endpoints
# ✅ Whisper transcription (MediaRecorder -> /api/transcribe)
#
# Render env vars you MUST set:
# - FLASK_SECRET_KEY (or APP_SECRET_KEY)
# - GOOGLE_CLIENT_ID
# - DEEPSEEK_API_KEY
# - STRIPE_SECRET_KEY (sk_live_...)
# - STRIPE_WEBHOOK_SECRET (whsec_...)
# - STRIPE_PRICE_ID_PRO=price_1T3yFlHvfV7kt9wle2LpbpU0
# - APP_BASE_URL=https://www.vividmedi.com

import os
import re
import sqlite3
import tempfile
import threading
import subprocess
from uuid import uuid4
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import stripe
from flask import Flask, request, jsonify, render_template, session, make_response
from faster_whisper import WhisperModel

# Google ID token verification (server-side)
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

# Bearer token signing
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired


# -----------------------------------
# Load .env locally only (not Render)
# -----------------------------------
if os.getenv("RENDER") is None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


# -----------------------------------
# Config
# -----------------------------------
APP_SECRET_KEY = (os.getenv("APP_SECRET_KEY") or os.getenv("FLASK_SECRET_KEY") or "dev-secret-change-me").strip()
GOOGLE_CLIENT_ID = (os.getenv("GOOGLE_CLIENT_ID") or "").strip()

DEEPSEEK_API_KEY = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
DEEPSEEK_MODEL = (os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
DEEPSEEK_URL = (os.getenv("DEEPSEEK_URL") or "https://api.deepseek.com/v1/chat/completions").strip()

# Runtime tuning (accuracy/speed)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_COMPUTE_TYPE = (os.getenv("WHISPER_COMPUTE_TYPE") or "int8").strip()
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "8"))
WHISPER_BEST_OF = int(os.getenv("WHISPER_BEST_OF", "5"))
WHISPER_LANGUAGE = (os.getenv("WHISPER_LANGUAGE") or "en").strip() or "en"
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0"))
WHISPER_INITIAL_PROMPT = (
    os.getenv("WHISPER_INITIAL_PROMPT")
    or "Australian clinical dictation. Medical terminology may include: DVA, HbA1c, COPD, AF, eGFR, CRP, troponin, metoprolol, apixaban."
)

DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "1200"))
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.15"))

STRIPE_SECRET_KEY = (os.getenv("STRIPE_SECRET_KEY") or "").strip()
STRIPE_WEBHOOK_SECRET = (os.getenv("STRIPE_WEBHOOK_SECRET") or "").strip()
STRIPE_PRICE_ID_PRO = (os.getenv("STRIPE_PRICE_ID_PRO") or "").strip()  # MUST be price_...
APP_BASE_URL = (os.getenv("APP_BASE_URL") or "https://www.vividmedi.com").rstrip("/")

# Optional: auto-pro for your own account
CREATOR_EMAIL = (os.getenv("CREATOR_EMAIL") or "").strip().lower()

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# HARD GUARD: prevents your old error (plink_ or URL passed as price)
if STRIPE_PRICE_ID_PRO:
    if STRIPE_PRICE_ID_PRO.startswith("plink_") or STRIPE_PRICE_ID_PRO.startswith("http"):
        raise RuntimeError(f"STRIPE_PRICE_ID_PRO must be a price_ id, not a payment link or URL: {STRIPE_PRICE_ID_PRO}")
    if not STRIPE_PRICE_ID_PRO.startswith("price_"):
        raise RuntimeError(f"STRIPE_PRICE_ID_PRO must start with 'price_': {STRIPE_PRICE_ID_PRO}")


# -----------------------------------
# Flask app
# -----------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = (os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or APP_SECRET_KEY or "dev-insecure-change-me")


# -----------------------------------
# DB (SQLite)
# -----------------------------------
DB_PATH = os.getenv("DB_PATH") or "vividmedi.db"

def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    with db_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            picture TEXT,
            plan TEXT NOT NULL DEFAULT 'free',         -- free | pro
            email_verified INTEGER NOT NULL DEFAULT 1,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            created_at TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS usage (
            actor_type TEXT NOT NULL,   -- guest | user
            actor_id TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (actor_type, actor_id)
        )
        """)
        conn.commit()

db_init()


# -----------------------------------
# Token auth helpers (Bearer)
# -----------------------------------
serializer = URLSafeTimedSerializer(APP_SECRET_KEY, salt="vm-auth")

def sign_token(user_id: str) -> str:
    return serializer.dumps({"uid": user_id})

def verify_token(token: str, max_age_seconds: int = 60 * 60 * 24 * 30):
    try:
        data = serializer.loads(token, max_age=max_age_seconds)
        return data.get("uid")
    except (BadSignature, SignatureExpired):
        return None

def get_user_from_bearer():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    uid = verify_token(token)
    if not uid:
        return None
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None

def get_authed_user():
    # Prefer Bearer (frontend apiFetch), fallback to session cookie
    u = get_user_from_bearer()
    if u:
        return u
    uid = session.get("user_id")
    if not uid:
        return None
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None


# -----------------------------------
# Guest cookie (quota tracking)
# -----------------------------------
def get_guest_id():
    return request.cookies.get("vivid_guest") or ""

def ensure_guest_cookie(resp):
    gid = get_guest_id()
    if not gid:
        gid = str(uuid4())
        resp.set_cookie(
            "vivid_guest",
            gid,
            httponly=True,
            secure=True,
            samesite="Lax",
            max_age=60 * 60 * 24 * 365,
            path="/",
        )
    return gid


# -----------------------------------
# Time helpers
# -----------------------------------
def now_awst() -> str:
    dt = datetime.now(ZoneInfo("Australia/Perth"))
    return dt.strftime("%d %b %Y, %H:%M (AWST)")


# -----------------------------------
# DVA header helpers
# -----------------------------------
def extract_field(text: str, labels):
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
    header.append(f"Date & Time (AWST): {now_awst()}")
    return "\n".join(header).strip()


# -----------------------------------
# User DB helpers
# -----------------------------------
def create_or_get_user_by_email(email: str, name: str = "", picture: str = "") -> dict:
    email = (email or "").strip().lower()
    if not email:
        raise ValueError("Missing email")

    with db_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if row:
            conn.execute(
                "UPDATE users SET name=?, picture=? WHERE email=?",
                (name or row["name"], picture or row["picture"], email),
            )
            conn.commit()
            row2 = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            return dict(row2)

        uid = "usr_" + uuid4().hex
        conn.execute(
            """
            INSERT INTO users (id, email, name, picture, plan, email_verified, created_at)
            VALUES (?, ?, ?, ?, 'free', 1, ?)
            """,
            (uid, email, name, picture, datetime.utcnow().isoformat()),
        )
        conn.commit()
        row2 = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        return dict(row2)

def upgrade_user_to_pro(user_id: str, stripe_customer_id: str = None, stripe_subscription_id: str = None):
    with db_conn() as conn:
        conn.execute(
            """
            UPDATE users
            SET plan='pro',
                stripe_customer_id=COALESCE(?, stripe_customer_id),
                stripe_subscription_id=COALESCE(?, stripe_subscription_id)
            WHERE id=?
            """,
            (stripe_customer_id, stripe_subscription_id, user_id),
        )
        conn.commit()


# -----------------------------------
# Usage / quota
# -----------------------------------
def usage_get(actor_type: str, actor_id: str) -> int:
    if not actor_id:
        return 0
    with db_conn() as conn:
        row = conn.execute(
            "SELECT used FROM usage WHERE actor_type=? AND actor_id=?",
            (actor_type, actor_id),
        ).fetchone()
        return int(row["used"]) if row else 0

def usage_incr(actor_type: str, actor_id: str, by: int = 1) -> int:
    if not actor_id:
        return 0
    with db_conn() as conn:
        row = conn.execute(
            "SELECT used FROM usage WHERE actor_type=? AND actor_id=?",
            (actor_type, actor_id),
        ).fetchone()
        if row:
            used = int(row["used"]) + by
            conn.execute(
                "UPDATE usage SET used=?, updated_at=? WHERE actor_type=? AND actor_id=?",
                (used, datetime.utcnow().isoformat(), actor_type, actor_id),
            )
        else:
            used = by
            conn.execute(
                "INSERT INTO usage (actor_type, actor_id, used, updated_at) VALUES (?, ?, ?, ?)",
                (actor_type, actor_id, used, datetime.utcnow().isoformat()),
            )
        conn.commit()
        return used

def actor_and_limit():
    """
    Guest: 10 total generations
    Logged-in free: 11 total generations
    Pro: effectively unlimited
    """
    u = get_authed_user()
    if u:
        actor_type = "user"
        actor_id = u["id"]
        limit = 1_000_000 if (u.get("plan") or "free") == "pro" else 11
        return actor_type, actor_id, limit, u

    gid = get_guest_id()
    return "guest", gid, 10, None

def quota_block_payload(used: int, limit: int, is_logged_in: bool):
    payload = {
        "error": "quota_exceeded",
        "used": min(used, limit),
        "limit": limit,
        "headline": "Free limit reached",
        "copy": [
            f"You’ve used {min(used, limit)}/{limit} free generations.",
            "Upgrade to Pro for unlimited access.",
            "Pro includes higher limits, priority processing, and ongoing updates.",
        ],
        "cta": {
            "primary": {"label": "Upgrade to Pro", "action": "upgrade"},
            "secondary": {
                "label": "Create account" if not is_logged_in else "Account",
                "action": "signup" if not is_logged_in else "account",
            },
        },
    }
    if not is_logged_in:
        payload["promo"] = {"label": "Create a free account to unlock 1 extra generation today."}
    return payload

def enforce_quota_or_402():
    actor_type, actor_id, limit, u = actor_and_limit()
    used_after = usage_incr(actor_type, actor_id, 1)
    if used_after > limit:
        return jsonify(quota_block_payload(used_after, limit, is_logged_in=bool(u))), 402
    return None


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
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
    return _whisper_model


# -----------------------------------
# Prompts
# -----------------------------------
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
    "Use Australian spelling.\n\n"
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


CLINICAL_PROMPT_APPEND = {
    "clinical": "General clinical reasoning mode.",
    "differential": (
        "Prioritise broad differential diagnosis: rank likely and dangerous causes, "
        "state supporting/opposing features for each, and include red-flag exclusion logic."
    ),
    "medication_review": (
        "Focus on medication safety and optimisation: interactions, duplications, contraindications, "
        "deprescribing opportunities, monitoring, and practical regimen simplification."
    ),
    "investigation_plan": (
        "Focus on pragmatic investigation strategy: first-line tests, escalation triggers, "
        "and how each result changes management."
    ),
}

CONSULT_PROMPT_APPEND = {
    "consult_note": "Build a polished consultation note from raw dictation.",
    "handover": "Build a concise, verbal-ready clinical handover.",
    "discharge_summary": (
        "Build a concise discharge summary with diagnosis, treatment provided, "
        "medication changes, pending tests, and explicit follow-up instructions."
    ),
}

# -----------------------------------
# Mode-aware prompt builders
# -----------------------------------
def build_generate_prompt(mode: str, query: str):
    if mode.startswith("dva"):
        header = build_dva_header(query)
        referral_intent = "D0904 new" if mode == "dva_new" else "D0904 renewal" if mode == "dva_renew" else "D0904 (unspecified)"
        user_content = (
            f"Referral intent: {referral_intent}\n\n"
            f"{header}\n\n"
            f"DETAILS:\n{query}\n\n"
            "Follow DVA_META format then clinical headings."
        )
        return DVA_SYSTEM_PROMPT, user_content

    mode_guidance = CLINICAL_PROMPT_APPEND.get(mode, CLINICAL_PROMPT_APPEND["clinical"])
    user_content = (
        f"Mode: {mode}\n"
        f"Guidance: {mode_guidance}\n\n"
        f"Clinical question:\n{query}\n\n"
        "If pasted data is included, sort it into the correct headings."
    )
    return CLINICAL_SYSTEM_PROMPT, user_content


def build_consult_prompt(mode: str, text: str):
    guidance = CONSULT_PROMPT_APPEND.get(mode, CONSULT_PROMPT_APPEND["consult_note"])

    if mode == "handover":
        user_content = (
            f"Guidance: {guidance}\n"
            "Create a handover/presentation from the following raw dictation/pasted data. "
            "If the context is not ED, adapt appropriately.\n\n"
            f"{text}"
        )
        return HANDOVER_SYSTEM_PROMPT, user_content

    if mode == "discharge_summary":
        user_content = (
            f"Guidance: {guidance}\n"
            "Create a discharge summary from the following raw dictation/pasted data. "
            "Do not invent facts. Ensure medication changes and follow-up actions are explicit.\n\n"
            f"{text}"
        )
        return CONSULT_NOTE_SYSTEM_PROMPT, user_content

    user_content = (
        f"Guidance: {guidance}\n"
        "Create a structured clinical note from the following raw dictation/pasted data. "
        "Do not invent facts; organise clearly.\n\n"
        f"{text}"
    )
    return CONSULT_NOTE_SYSTEM_PROMPT, user_content


# -----------------------------------
# DeepSeek call
# -----------------------------------
http = requests.Session()

def call_deepseek(system_prompt: str, user_content: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": DEEPSEEK_TEMPERATURE,
        "top_p": 0.9,
        "max_tokens": DEEPSEEK_MAX_TOKENS,
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


# -----------------------------------
# Health checks
# -----------------------------------
@app.get("/health")
def health():
    return "ok", 200

@app.get("/healthz")
def healthz():
    return "ok", 200

@app.get("/_ping")
def ping():
    return "pong", 200


# -----------------------------------
# Pages
# -----------------------------------
@app.get("/")
def index():
    resp = make_response(render_template("index.html", google_client_id=GOOGLE_CLIENT_ID))
    ensure_guest_cookie(resp)
    return resp

@app.get("/pro/success")
def pro_success():
    resp = make_response(render_template("index.html", google_client_id=GOOGLE_CLIENT_ID))
    ensure_guest_cookie(resp)
    return resp

@app.get("/pro/cancelled")
def pro_cancelled():
    resp = make_response(render_template("index.html", google_client_id=GOOGLE_CLIENT_ID))
    ensure_guest_cookie(resp)
    return resp


# -----------------------------------
# Session utility endpoint
# -----------------------------------
@app.get("/api/session")
def api_session():
    resp = make_response(jsonify({"ok": True}))
    ensure_guest_cookie(resp)
    return resp


# -----------------------------------
# Me endpoint
# -----------------------------------
@app.get("/api/me")
def api_me():
    u = get_authed_user()
    actor_type, actor_id, limit, _u = actor_and_limit()
    used = usage_get(actor_type, actor_id)

    return jsonify({
        "logged_in": bool(u),
        "email": u["email"] if u else None,
        "plan": u["plan"] if u else "guest",
        "used": used,
        "limit": limit,
        "remaining": max(0, limit - used),
    })


# -----------------------------------
# Google auth
# -----------------------------------
@app.post("/auth/google")
def auth_google():
    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "Server misconfigured: missing GOOGLE_CLIENT_ID"}), 500

    data = request.get_json(silent=True) or {}
    token = (data.get("credential") or "").strip()
    if not token:
        return jsonify({"error": "Missing credential"}), 400

    try:
        info = google_id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        email = (info.get("email") or "").strip().lower()
        name = (info.get("name") or "") or (info.get("given_name") or "")
        picture = (info.get("picture") or "")

        user = create_or_get_user_by_email(email=email, name=name, picture=picture)

        # Optional: auto-pro for creator email
        if CREATOR_EMAIL and email == CREATOR_EMAIL and user.get("plan") != "pro":
            upgrade_user_to_pro(user["id"])
            user["plan"] = "pro"

        session["user_id"] = user["id"]
        bearer = sign_token(user["id"])

        return jsonify({
            "ok": True,
            "token": bearer,  # ✅ index.html uses this
            "user": {"email": user["email"], "plan": user["plan"]},
        })

    except Exception as e:
        print("GOOGLE AUTH ERROR:", repr(e))
        return jsonify({"error": "Google sign-in failed"}), 401


@app.post("/auth/logout")
def auth_logout():
    session.clear()
    return jsonify({"ok": True})


# -----------------------------------
# Stripe checkout (SUBSCRIPTION)
# -----------------------------------
@app.post("/api/stripe/create-checkout-session")
def stripe_create_checkout_session():
    if not STRIPE_SECRET_KEY:
        return jsonify({"error": "Server misconfigured: missing STRIPE_SECRET_KEY"}), 500
    if not STRIPE_PRICE_ID_PRO:
        return jsonify({"error": "Server misconfigured: missing STRIPE_PRICE_ID_PRO"}), 500

    u = get_authed_user()
    if not u:
        return jsonify({"error": "not_authenticated"}), 401

    try:
        customer_id = u.get("stripe_customer_id")

        # Create Stripe customer ONCE
        if not customer_id:
            customer = stripe.Customer.create(
                email=u["email"],
                metadata={"user_id": u["id"]},
            )
            customer_id = customer.id
            with db_conn() as conn:
                conn.execute("UPDATE users SET stripe_customer_id=? WHERE id=?", (customer_id, u["id"]))
                conn.commit()

        # Create subscription checkout session using PRICE id
        sess = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": STRIPE_PRICE_ID_PRO, "quantity": 1}],
            success_url=f"{APP_BASE_URL}/pro/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{APP_BASE_URL}/pro/cancelled",
            metadata={"user_id": u["id"], "product": "vividmedi_pro"},
        )

        return jsonify({"url": sess.url})

    except Exception as e:
        print("STRIPE CHECKOUT ERROR:", repr(e))
        return jsonify({"error": str(e)}), 500


# -----------------------------------
# Stripe webhook — upgrades user via stripe_customer_id
# -----------------------------------
@app.post("/api/stripe/webhook")
def stripe_webhook():
    if not STRIPE_WEBHOOK_SECRET:
        return "Missing webhook secret", 500

    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        print("STRIPE WEBHOOK SIGNATURE ERROR:", repr(e))
        return "Bad signature", 400

    etype = event.get("type", "")
    obj = (event.get("data") or {}).get("object") or {}

    # Upgrade user when checkout completes
    if etype == "checkout.session.completed":
        customer_id = obj.get("customer")
        subscription_id = obj.get("subscription")

        if customer_id:
            with db_conn() as conn:
                row = conn.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,)).fetchone()
            if row:
                upgrade_user_to_pro(
                    user_id=row["id"],
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=subscription_id,
                )

    # Optional: downgrade on cancellation/unpaid
    if etype in ("customer.subscription.deleted", "customer.subscription.updated"):
        sub = obj
        customer_id = sub.get("customer")
        status = (sub.get("status") or "").lower()
        if customer_id:
            with db_conn() as conn:
                row = conn.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,)).fetchone()
                if row and status in ("canceled", "unpaid", "incomplete_expired"):
                    conn.execute("UPDATE users SET plan='free' WHERE id=?", (row["id"],))
                    conn.commit()

    return "OK", 200


# -----------------------------------
# AI endpoints
# -----------------------------------
@app.post("/api/generate")
def generate():
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    mode = (data.get("mode") or "clinical").strip().lower()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    blocked = enforce_quota_or_402()
    if blocked:
        return blocked

    try:
        system_prompt, user_content = build_generate_prompt(mode, query)
        answer = call_deepseek(system_prompt, user_content)

        return jsonify({"answer": answer})

    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "AI request failed"}), 502


@app.post("/api/consult")
def consult():
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "Server misconfigured: missing DEEPSEEK_API_KEY"}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    mode = (data.get("mode") or "consult_note").strip().lower()

    if not text:
        return jsonify({"error": "Empty input"}), 400

    blocked = enforce_quota_or_402()
    if blocked:
        return blocked

    try:
        system_prompt, user_content = build_consult_prompt(mode, text)
        answer = call_deepseek(system_prompt, user_content)

        return jsonify({"answer": answer})

    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "AI request failed"}), 502


# -----------------------------------
# Transcribe endpoint
# -----------------------------------
@app.post("/api/transcribe")
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
            segments, _info = model.transcribe(
                wav_path,
                beam_size=WHISPER_BEAM_SIZE,
                best_of=WHISPER_BEST_OF,
                language=WHISPER_LANGUAGE,
                temperature=WHISPER_TEMPERATURE,
                initial_prompt=WHISPER_INITIAL_PROMPT,
                vad_filter=True,
            )

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


# -----------------------------------
# Local run
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
