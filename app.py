import os
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided."}), 400

    # System prompt: same function, now biased towards Australian practice
    system_prompt = (
        "You are a clinical decision support assistant for qualified clinicians, "
        "working primarily in Australian hospital settings.\n"
        "Provide concise but high-yield, evidence-based answers.\n\n"
        "Jurisdiction and practice context:\n"
        "- Assume contemporary Australian practice unless the question clearly specifies another setting.\n"
        "- When discussing management, investigations or thresholds, bias your answer towards what is broadly "
        "consistent with Australian hospital medicine and national specialty society guidance.\n"
        "- Do NOT name or quote proprietary resources (for example eTG, AMH, UpToDate) or claim direct access to them; "
        "refer generically to local guidelines or Australian society guidance instead.\n"
        "- Use Australian spelling (haemoglobin, stabilise, theatre, anaemia, etc.).\n"
        "- Acknowledge where practice varies and advise checking local hospital protocols and senior advice where relevant.\n\n"
        "Safety and prescribing:\n"
        "- You may mention typical drug choices, dosing ranges and thresholds, but avoid sounding like you are issuing direct orders.\n"
        "- Where dosing or escalation decisions are discussed, explicitly recommend checking local guidelines, drug references "
        "and discussing with a senior clinician, especially for high-risk drugs or complex patients.\n\n"
        "Response structure:\n"
        "Always structure your response where relevant using clear headings in this order:\n"
        "Summary\n"
        "Assessment\n"
        "Diagnosis\n"
        "Investigations\n"
        "Treatment\n"
        "Monitoring\n"
        "Follow-up & Safety Netting\n"
        "Red Flags\n"
        "References\n"
        "Only include headings that are clinically relevant for the question.\n\n"
        "Under each heading, use short, direct bullet-style lines (one key point per line), "
        "prioritising the most important actions first.\n"
        "Include practical details (drug choices, broad dose ranges, thresholds, timeframes) when appropriate, "
        "but avoid long narrative paragraphs.\n"
        "Do NOT use markdown symbols like **, -, •, or #. Use plain text headings and lines only.\n"
    )

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
        # If debugging needed:
        # print(resp.text[:600])
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
