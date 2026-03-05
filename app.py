# app.py — VividMedi with TRULY INTELLIGENT AI (Query Classification + Adaptive Prompts)
# Now actually smart: detects simple vs complex questions, responds conversationally

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
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired


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
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
STRIPE_SECRET_KEY = (os.getenv("STRIPE_SECRET_KEY") or "").strip()
STRIPE_WEBHOOK_SECRET = (os.getenv("STRIPE_WEBHOOK_SECRET") or "").strip()
STRIPE_PRICE_ID_PRO = (os.getenv("STRIPE_PRICE_ID_PRO") or "").strip()
APP_BASE_URL = (os.getenv("APP_BASE_URL") or "https://www.vividmedi.com").rstrip("/")
CREATOR_EMAIL = (os.getenv("CREATOR_EMAIL") or "").strip().lower()

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

if STRIPE_PRICE_ID_PRO:
    if STRIPE_PRICE_ID_PRO.startswith("plink_") or STRIPE_PRICE_ID_PRO.startswith("http"):
        raise RuntimeError(f"STRIPE_PRICE_ID_PRO must be a price_ id: {STRIPE_PRICE_ID_PRO}")
    if not STRIPE_PRICE_ID_PRO.startswith("price_"):
        raise RuntimeError(f"STRIPE_PRICE_ID_PRO must start with 'price_': {STRIPE_PRICE_ID_PRO}")


app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = (os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or APP_SECRET_KEY or "dev-insecure-change-me")

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=APP_BASE_URL.startswith("https://"),
)


# -----------------------------------
# DB (SQLite)
# -----------------------------------
DB_PATH = os.getenv("DB_PATH") or "vividmedi.db"


def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def db_init():
    with db_conn() as conn:
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            picture TEXT,
            plan TEXT NOT NULL DEFAULT 'free',
            email_verified INTEGER NOT NULL DEFAULT 1,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            specialty TEXT,
            expertise_level TEXT,
            created_at TEXT NOT NULL
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS usage (
            actor_type TEXT NOT NULL,
            actor_id TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (actor_type, actor_id)
        )
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            mode TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
        )
        conn.commit()


db_init()


# -----------------------------------
# Token auth helpers
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
    u = get_user_from_bearer()
    if u:
        return u
    uid = session.get("user_id")
    if not uid:
        return None
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None


def get_guest_id():
    return request.cookies.get("vivid_guest") or ""


def ensure_guest_cookie(resp):
    gid = get_guest_id()
    if not gid:
        gid = str(uuid4())
        resp.set_cookie(
            "vivid_guest", gid,
            httponly=True,
            secure=APP_BASE_URL.startswith("https://"),
            samesite="Lax",
            max_age=60 * 60 * 24 * 365,
            path="/",
        )
    return gid


def now_awst() -> str:
    dt = datetime.now(ZoneInfo("Australia/Perth"))
    return dt.strftime("%d %b %Y, %H:%M (AWST)")


# -----------------------------------
# INTELLIGENT QUERY CLASSIFICATION
# -----------------------------------
def classify_query(query: str) -> str:
    """Intelligently detect query type to determine response format"""
    query_lower = query.lower()
    query_len = len(query)
    
    # Factual: dosing, definitions, side effects
    if any(x in query_lower for x in ['dose', 'dosage', 'how much', 'concentration', 'mg', 'frequency', 'define', 'what is']):
        return 'factual'
    
    # Simple/follow-up: short, contextual
    if query_len < 100 and any(x in query_lower for x in ['what about', 'and', 'also', 'if', 'alternatives']):
        return 'followup'
    
    # Complex clinical case: long, patient-centered
    if query_len > 500 or any(x in query_lower for x in ['patient', 'case', 'presentation', 'presenting', 'history of', 'exam']):
        return 'complex'
    
    # Decision support: management questions
    if any(x in query_lower for x in ['should i', 'next step', 'manage', 'treat', 'investigate', 'workup', 'assessment']):
        return 'clinical'
    
    return 'clinical'


# -----------------------------------
# ADAPTIVE SYSTEM PROMPTS (not template-based)
# -----------------------------------

PROMPT_FACTUAL = """You are a clinical reference assistant for Australian doctors.

TASK: Answer factual questions directly and concisely.
- Answer in 1-2 sentences max
- Reference Australian TGA product info where relevant
- Include key contraindications or interactions
- NO headers, NO structure, NO templates
- Just the answer

Example Q: What's the dose of amoxicillin for UTI?
Example A: Amoxicillin 500mg 8-hourly or 1g 12-hourly for 3-5 days for uncomplicated UTI. Reduce dose in renal impairment; contraindicated in penicillin allergy."""

PROMPT_FOLLOWUP = """You are a clinical assistant in ongoing conversation.

TASK: Answer the specific follow-up question asked.
- Be brief (1-2 sentences)
- Build on context from previous questions
- NO headers, structure, or templates
- Just answer the question directly

You're having a ward round conversation, not writing a report."""

PROMPT_CLINICAL = """You are an expert Australian clinician providing clinical decision support.

TASK: Give actionable clinical guidance.

Structure ONLY if helpful:
1. Scenario summary (1 sentence)
2. Key considerations (3-4 bullet points)
3. Recommended approach with evidence
4. Safety netting (what could go wrong)

If a simple answer suffices, give it directly. Avoid unnecessary structure.
Think like a consultant, not a form-filler."""

PROMPT_COMPLEX = """You are an expert Australian clinical advisor for complex patient cases.

TASK: Provide comprehensive clinical analysis.

Your reasoning:
1. RED FLAGS FIRST: Identify urgent/dangerous features immediately
2. DIFFERENTIAL DIAGNOSIS: 3-5 most likely with probability and distinguishing features
3. INVESTIGATIONS: Prioritized by urgency and diagnostic value
4. MANAGEMENT: Evidence-based, Australian guidelines (TGA, NHMRC, Therapeutic Guidelines)
5. MONITORING: Specific triggers for escalation or review

Output naturally. Use structure only where it aids clarity. 
Be specific about risk stratification and patient factors."""

PROMPT_DVA = """You are an expert Australian medical practitioner for DVA D0904 referrals.

TASK: Provide detailed DVA audit analysis.

Your analysis:
1. REFERRAL TYPE: New, renewal, or EoC-based?
2. PROVIDER FIT: Does provider type match DVA-accepted disciplines?
3. JUSTIFICATION STRENGTH: How defensible is this referral in audit?
4. AUDIT RISK: What elements could trigger denial?
5. ALTERNATIVES: Legitimate pathways if weak?

Output format:
DVA_META
Referral type: [type]
Provider type: [discipline]
Justification strength: [strong|moderate|weak]
Audit risk: [low|medium|high]
[Continue with bullet-point findings]
END_DVA_META

Then provide clinical sections if needed.

IMPORTANT: Do not invent entitlements. Do not advise misrepresentation."""


# -----------------------------------
# Conversation context
# -----------------------------------
def get_conversation_context(user_id: str, limit: int = 3) -> str:
    """Get recent conversation for awareness (not forcing context)"""
    if not user_id:
        return ""
    
    with db_conn() as conn:
        rows = conn.execute(
            "SELECT query FROM conversation_history WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    
    if not rows:
        return ""
    
    context = "Context from previous questions:\n"
    for row in reversed(rows):
        q = (row["query"] or "")[:80]
        context += f"- {q}\n"
    
    return context


def save_conversation(user_id: str, query: str, answer: str, mode: str = "clinical"):
    if not user_id:
        return
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO conversation_history (user_id, query, answer, mode, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, query, answer, mode, datetime.utcnow().isoformat()),
        )
        conn.commit()


# -----------------------------------
# DeepSeek call
# -----------------------------------
http = requests.Session()


def call_deepseek(system_prompt: str, user_content: str, context: str = "") -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")

    full_content = user_content
    if context:
        full_content = f"{context}\n\n{user_content}"

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_content},
        ],
        "temperature": 0.4,  # Balanced for reasoning + naturalness
        "top_p": 0.9,
        "max_tokens": 1500,
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
# Health / Pages / Auth (same as before)
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


@app.get("/api/session")
def api_session():
    resp = make_response(jsonify({"ok": True}))
    ensure_guest_cookie(resp)
    return resp


@app.get("/api/me")
def api_me():
    u = get_authed_user()
    actor_type, actor_id, limit, _u = actor_and_limit()
    used = usage_get(actor_type, actor_id)

    return jsonify(
        {
            "logged_in": bool(u),
            "email": u["email"] if u else None,
            "plan": u["plan"] if u else "guest",
            "used": used,
            "limit": limit,
            "remaining": max(0, limit - used),
        }
    )


# User/quota functions (same as before)
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
            "INSERT INTO users (id, email, name, picture, plan, email_verified, created_at) VALUES (?, ?, ?, ?, 'free', 1, ?)",
            (uid, email, name, picture, datetime.utcnow().isoformat()),
        )
        conn.commit()
        row2 = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        return dict(row2)


def upgrade_user_to_pro(user_id: str, stripe_customer_id: str = None, stripe_subscription_id: str = None):
    with db_conn() as conn:
        conn.execute(
            "UPDATE users SET plan='pro', stripe_customer_id=COALESCE(?, stripe_customer_id), stripe_subscription_id=COALESCE(?, stripe_subscription_id) WHERE id=?",
            (stripe_customer_id, stripe_subscription_id, user_id),
        )
        conn.commit()


def downgrade_user_to_free_by_customer(customer_id: str):
    with db_conn() as conn:
        row = conn.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,)).fetchone()
        if row:
            conn.execute("UPDATE users SET plan='free' WHERE id=?", (row["id"],))
            conn.commit()


def usage_get(actor_type: str, actor_id: str) -> int:
    if not actor_id:
        return 0
    with db_conn() as conn:
        row = conn.execute("SELECT used FROM usage WHERE actor_type=? AND actor_id=?", (actor_type, actor_id)).fetchone()
        return int(row["used"]) if row else 0


def usage_incr(actor_type: str, actor_id: str, by: int = 1) -> int:
    if not actor_id:
        return 0
    with db_conn() as conn:
        row = conn.execute("SELECT used FROM usage WHERE actor_type=? AND actor_id=?", (actor_type, actor_id)).fetchone()
        if row:
            used = int(row["used"]) + by
            conn.execute("UPDATE usage SET used=?, updated_at=? WHERE actor_type=? AND actor_id=?", (used, datetime.utcnow().isoformat(), actor_type, actor_id))
        else:
            used = by
            conn.execute("INSERT INTO usage (actor_type, actor_id, used, updated_at) VALUES (?, ?, ?, ?)", (actor_type, actor_id, used, datetime.utcnow().isoformat()))
        conn.commit()
        return used


def actor_and_limit():
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
            f"You've used {min(used, limit)}/{limit} free generations.",
            "Upgrade to Pro for unlimited access.",
        ],
    }
    return payload


def enforce_quota_or_402():
    actor_type, actor_id, limit, u = actor_and_limit()
    used_before = usage_get(actor_type, actor_id)
    if used_before >= limit:
        return jsonify(quota_block_payload(used_before, limit, is_logged_in=bool(u))), 402

    usage_incr(actor_type, actor_id, 1)
    return None


@app.post("/auth/google")
def auth_google():
    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "Server misconfigured: missing GOOGLE_CLIENT_ID"}), 500

    data = request.get_json(silent=True) or {}
    token = (data.get("credential") or "").strip()
    if not token:
        return jsonify({"error": "Missing credential"}), 400

    try:
        info = google_id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)

        email = (info.get("email") or "").strip().lower()
        name = (info.get("name") or "") or (info.get("given_name") or "")
        picture = (info.get("picture") or "")

        user = create_or_get_user_by_email(email=email, name=name, picture=picture)

        if CREATOR_EMAIL and email == CREATOR_EMAIL and user.get("plan") != "pro":
            upgrade_user_to_pro(user["id"])
            user["plan"] = "pro"

        session["user_id"] = user["id"]
        bearer = sign_token(user["id"])

        return jsonify({"ok": True, "token": bearer, "user": {"email": user["email"], "plan": user["plan"]}})

    except Exception as e:
        print("GOOGLE AUTH ERROR:", repr(e))
        return jsonify({"error": "Google sign-in failed"}), 401


@app.post("/auth/logout")
def auth_logout():
    session.clear()
    return jsonify({"ok": True})


# Stripe (same as before, abbreviated for space)
@app.post("/api/stripe/create-checkout-session")
def stripe_create_checkout_session():
    if not STRIPE_SECRET_KEY or not STRIPE_PRICE_ID_PRO:
        return jsonify({"error": "Server misconfigured"}), 500

    u = get_authed_user()
    if not u:
        return jsonify({"error": "not_authenticated"}), 401

    try:
        customer_id = u.get("stripe_customer_id")

        if not customer_id:
            customer = stripe.Customer.create(email=u["email"], metadata={"user_id": u["id"]})
            customer_id = customer.id
            with db_conn() as conn:
                conn.execute("UPDATE users SET stripe_customer_id=? WHERE id=?", (customer_id, u["id"]))
                conn.commit()

        sess = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": STRIPE_PRICE_ID_PRO, "quantity": 1}],
            success_url=f"{APP_BASE_URL}/pro/success?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{APP_BASE_URL}/pro/cancelled",
            metadata={"user_id": u["id"], "product": "vividmedi_pro"},
            client_reference_id=u["id"],
        )

        return jsonify({"url": sess.url})

    except Exception as e:
        print("STRIPE CHECKOUT ERROR:", repr(e))
        return jsonify({"error": str(e)}), 500


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

    if etype == "checkout.session.completed":
        customer_id = obj.get("customer")
        subscription_id = obj.get("subscription")
        if customer_id and subscription_id:
            with db_conn() as conn:
                row = conn.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,)).fetchone()
                if row:
                    conn.execute("UPDATE users SET stripe_subscription_id=? WHERE id=?", (subscription_id, row["id"]))
                    conn.commit()

    if etype == "invoice.paid":
        customer_id = obj.get("customer")
        subscription_id = obj.get("subscription")
        if customer_id:
            with db_conn() as conn:
                row = conn.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,)).fetchone()
            if row:
                upgrade_user_to_pro(row["id"], stripe_customer_id=customer_id, stripe_subscription_id=subscription_id)

    if etype in ("customer.subscription.created", "customer.subscription.updated"):
        sub = obj
        customer_id = sub.get("customer")
        status = (sub.get("status") or "").lower()
        sub_id = sub.get("id")
        if customer_id and status in ("active", "trialing"):
            with db_conn() as conn:
                row = conn.execute("SELECT id FROM users WHERE stripe_customer_id=?", (customer_id,)).fetchone()
            if row:
                upgrade_user_to_pro(row["id"], stripe_customer_id=customer_id, stripe_subscription_id=sub_id)

        if customer_id and status in ("canceled", "unpaid", "incomplete_expired"):
            downgrade_user_to_free_by_customer(customer_id)

    if etype == "customer.subscription.deleted":
        customer_id = obj.get("customer")
        if customer_id:
            downgrade_user_to_free_by_customer(customer_id)

    return "OK", 200


# -----------------------------------
# AI endpoints with INTELLIGENT classification
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
        u = get_authed_user()
        user_id = u["id"] if u else None
        context = get_conversation_context(user_id) if user_id else ""

        if mode.startswith("dva"):
            system_prompt = PROMPT_DVA
            user_content = f"DVA referral analysis:\n\n{query}"
        else:
            # INTELLIGENT classification
            query_type = classify_query(query)
            
            if query_type == 'factual':
                system_prompt = PROMPT_FACTUAL
            elif query_type == 'followup':
                system_prompt = PROMPT_FOLLOWUP
            elif query_type == 'complex':
                system_prompt = PROMPT_COMPLEX
            else:
                system_prompt = PROMPT_CLINICAL
            
            user_content = query

        answer = call_deepseek(system_prompt, user_content, context)

        if user_id:
            save_conversation(user_id, query, answer, mode)

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
        u = get_authed_user()
        user_id = u["id"] if u else None
        context = get_conversation_context(user_id) if user_id else ""

        # For consult, use COMPLEX prompt (detailed output needed)
        system_prompt = PROMPT_COMPLEX
        user_content = f"Create a clinical note from raw dictation:\n\n{text}"

        answer = call_deepseek(system_prompt, user_content, context)

        if user_id:
            save_conversation(user_id, text[:100], answer, mode)

        return jsonify({"answer": answer})

    except Exception as e:
        print("DEEPSEEK ERROR:", repr(e))
        return jsonify({"error": "AI request failed"}), 502


# Transcribe (same as before)
_whisper_model = None
_whisper_init_lock = threading.Lock()
_transcribe_lock = threading.Lock()


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_init_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                _whisper_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device="cpu",
                    compute_type="int8",
                )
    return _whisper_model


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
