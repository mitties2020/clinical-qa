import os
import time
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")

app = Flask(__name__, template_folder="templates", static_folder="static")
session = requests.Session()


# -----------------------------------------------------------
# TRAFFIC LOGGING
# -----------------------------------------------------------
@app.before_request
def log_request_info():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    ua = request.headers.get("User-Agent", "unknown")
    path = request.path
    method = request.method

    print(f"[TRACK] {ip} {method} {path} {ua}")


# -----------------------------------------------------------
# ROUTES
# -----------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided."}), 400

    # -----------------------------------------------------------
    # SYSTEM PROMPT – Australian-biased medical practice
    # -----------------------------------------------------------
    system_prompt = (
        "You are a clinical decision support assistant for qualified clinicians, "
        "working primarily in Australian hospital settings.\n"
        "Provide concise, high-yield, evidence-based answers.\n\n"

        "Australian practice context:\n"
        "- Assume Australian hospital medicine unless stated otherwise.\n"
        "- Bias recommendations toward contemporary Australian practice and commonly accepted guidance "
        "from national specialty societies and hospital protocols.\n"
        "- Use Australian spelling (haemoglobin, stabilise, anaemia, theatre, etc.).\n"
        "- Do NOT cite proprietary sources such as eTG, AMH, or UpToDate. Refer instead to "
        "'local hospital guidelines' or 'Australian society recommendations'.\n\n"

        "Safety:\n"
        "- You may discuss medication choices and typical dose ranges, but avoid sounding like you are "
        "issuing orders. Encourage checking of local guidelines, renal dosing requirements, and senior review.\n"
        "- Acknowledge uncertainty where practice varies.\n\n"

        "Structure:\n"
        "Always structure responses (where clinically relevant) in this order:\n"
        "Summary\n"
        "Assessment\n"
        "Diagnosis\n"
        "Investigations\n"
        "Treatment\n"
        "Monitoring\n"
        "Follow-up & Safety Netting\n"
        "Red Flags\n"
        "References\n"
        "Only include headings that make sense for the question.\n\n"

        "Formatting:\n"
        "- Use short, direct, bullet-style statements (one key fact per line).\n"
        "- No markdown symbols: do NOT use **, -, •, or #.\n"
        "- Use plain text headings exactly as listed above.\n"
    )

    # -----------------------------------------------------------
    # DeepSeek payload
    # -----------------------------------------------------------
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
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


# -----------------------------------------------------------
# RUN APP
# -----------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
