import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# =========================
# Environment / DeepSeek
# =========================
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")

app = Flask(__name__, template_folder="templates", static_folder="static")


# Optional: simple traffic logging
@app.before_request
def log_request_info():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    ua = request.headers.get("User-Agent", "unknown")
    path = request.path
    method = request.method
    print(f"[TRACK] {ip} {method} {path} {ua}")


def call_deepseek(messages):
    """Call DeepSeek chat completion API."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
    }
    resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


@app.route("/")
def index():
    # Single-page app – just render index.html
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400

    system_prompt = (
        "You are an Australian clinical education assistant. "
        "You provide evidence-based, guideline-aware information using Australian spelling, "
        "but you do not give individual treatment instructions or dosing. "
        "You always remind users that your answers are general educational information only "
        "and not a substitute for local guidelines or senior clinician review."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        answer = call_deepseek(messages)
    except Exception as e:
        print(f"[ERROR] DeepSeek API failed: {e}")
        return jsonify({"error": "Backend error contacting model."}), 500

    return jsonify({"answer": answer})


@app.route("/ping")
def ping():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
