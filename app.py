import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

if not API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/clinical-qa", methods=["POST"])
def clinical_qa():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "No clinical question received."}), 400

    system_prompt = (
        "You are a structured, evidence-based clinical Q&A assistant for doctors in Australia. "
        "Provide concise, high-yield answers with headings: Overview, Assessment, Risk Stratification, "
        "Management, Monitoring, Red Flags, and Key References. "
        "End with 'Always verify with local policies and senior review.'"
    )

    try:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.3,
            "max_tokens": 900,
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)

        if resp.status_code != 200:
            # Return whatever DeepSeek said so we can see it in the browser
            return jsonify({
                "error": "DeepSeek API error",
                "status": resp.status_code,
                "body": resp.text,
            }), 500

        data = resp.json()
        # DeepSeek is OpenAI-compatible: choices[0].message.content
        answer = data["choices"][0]["message"]["content"].strip()
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": "Server exception", "details": str(e)}), 500

if __name__ == "__main__":
    # Use a fixed port; if it's busy, kill old processes: pkill -f app.py
    app.run(host="0.0.0.0", port=5000, debug=True)
