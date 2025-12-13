import os
import json
import time
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
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")  # base or small recommended

# Where to store learned corrections
# If you attach a Render Persistent Disk, set:
#   CORRECTIONS_PATH=/var/data/corrections.json
CORRECTIONS_PATH = os.getenv("CORRECTIONS_PATH", "corrections.json")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY")

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
# CORRECTIONS (LEARNING LAYER)
# --------------------------------------------------
_corrections_lock = threading.Lock()

def _load_corrections():
    if not os.path.exists(CORRECTIONS_PATH):
        return {"rules": []}
    try:
        with open(CORRECTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "rules" not in data:
            return {"rules": []}
        if not isinstance(data["rules"], list):
            data["rules"] = []
        return data
    except Exception:
        return {"rules": []}

def _save_corrections(data):
    tmp = CORRECTIONS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CORRECTIONS_PATH)

def _norm(s: str) -> str:
    return " ".join((s or "").lower().strip().split())

def _apply_rules(raw_text: str, rules):
    """
    Apply learned replacements to raw_text (case-insensitive).
    We keep it conservative: only replace full phrase occurrences.
    """
    text = raw_text
    low = raw_text.lower()
    for r in rules:
        wrong = (r.get("wrong") or "").strip()
        correct = (r.get("correct") or "").strip()
        if not wrong or not correct:
            continue
        # phrase match (case-insensitive)
        if wrong.lower() in low:
            # replace all occurrences, case-insensitive
            # simple safe approach: operate on lowercase copy then reassign
            low = low.replace(wrong.lower(), correct)
            text = low  # we return normalised lower text (consistent for queries)
    return text.strip()

def _derive_rule(old_raw: str, new_raw: str):
    """
    Create a rule mapping the older transcript (likely wrong) to the newer (likely correct).
    Conservative:
    - We only learn if both are non-empty and different.
    - We learn phrase-level replacement using the whole old -> new (works best for repeatable confusions).
    """
    o = _norm(old_raw)
    n = _norm(new_raw)
    if not o or not n:
        return None
    if o == n:
        return None

    # Heuristic: assume older is wrong, newer is correct (user repeated carefully)
    # Keep rule short to avoid over-broad replacements.
    # If old is very long, learning it as a whole phrase could be risky; cap it.
    if len(o) > 120 or len(n) > 160:
        # try to learn only the differing "core" by taking last 12 words
        o = " ".join(o.split()[-12:])
        n = " ".join(n.split()[-16:])

    return {"wrong": o, "correct": n, "count": 1, "last_seen": int(time.time())}

def _upsert_rule(data, rule):
    rules = data.get("rules", [])
    wrong = rule["wrong"]
    correct = rule["correct"]

    for r in rules:
        if _norm(r.get("wrong")) == wrong:
            # Update existing
            if _norm(r.get("correct")) == correct:
                r["count"] = int(r.get("count", 0)) + 1
                r["last_seen"] = int(time.time())
                return data
            else:
                # same wrong mapped to new correct -> overwrite to latest
                r["correct"] = correct
                r["count"] = int(r.get("count", 0)) + 1
                r["last_seen"] = int(time.time())
                return data

    rules.append(rule)
    data["rules"] = rules[-200:]  # cap growth
    return data

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# --------------------------------------------------
# TRANSCRIBE
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
            "Australian doctor dictation. Common terms: "
            "morphine, fentanyl, ketamine, ondansetron, metoclopramide, "
            "paracetamol, ibuprofen, ceftriaxone, amoxicillin, adrenaline, noradrenaline, "
            "lamotrigine, levetiracetam, sodium valproate, phenytoin, thiamine, Wernicke, "
            "NG tube, refeeding syndrome, phosphate, magnesium, "
            "dose, dosage, milligrams, micrograms, per kilogram, over 24 hours."
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

        with _corrections_lock:
            data = _load_corrections()
            rules = data.get("rules", [])
        corrected = _apply_rules(raw, rules)

        return jsonify({
            "raw": raw,
            "text": corrected,
            "language": getattr(info, "language", "en"),
            "rules_count": len(rules)
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
# LEARN CORRECTION (Incorrect → repeat)
# --------------------------------------------------
@app.route("/api/learn_correction", methods=["POST"])
def learn_correction():
    data_in = request.get_json(silent=True) or {}
    old_raw = data_in.get("old_raw") or ""
    new_raw = data_in.get("new_raw") or ""

    rule = _derive_rule(old_raw, new_raw)
    if not rule:
        return jsonify({"ok": False, "error": "Nothing to learn"}), 400

    with _corrections_lock:
        data = _load_corrections()
        data = _upsert_rule(data, rule)
        _save_corrections(data)
        rules = data.get("rules", [])

    # Also return the corrected "new" transcript after applying all rules (including the new one)
    corrected_new = _apply_rules(new_raw, rules)

    return jsonify({
        "ok": True,
        "learned": {"wrong": rule["wrong"], "correct": rule["correct"]},
        "rules_count": len(rules),
        "text": corrected_new,
        "raw": new_raw
    })

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
        "You are an AI clinical education assistant for Australian clinicians.\n\n"
        "Purpose and limits:\n"
        "Educational only; not real-time patient care.\n"
        "Use Australian spelling.\n"
        "Do not issue directives; encourage checking local guidelines.\n\n"
        "Structure responses using headings in order (only include relevant ones):\n"
        "Summary\nAssessment\nDiagnosis\nInvestigations\nTreatment\nMonitoring\n"
        "Follow-up & Safety Netting\nRed Flags\nReferences\n\n"
        "Formatting:\n"
        "Plain text headings exactly as written. One key point per line. No markdown symbols."
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
        out = r.json()
        answer = ((out.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
        if not answer:
            return jsonify({"error": "Empty response"}), 502
        return jsonify({"answer": answer})
    except Exception as e:
        print("[DEEPSEEK ERROR]", repr(e))
        return jsonify({"error": "LLM request failed"}), 502

# --------------------------------------------------
# MAIN (LOCAL DEV)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
