import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

if not API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY in .env")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    # Renders the modern UI
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}
    question = (data.get("query") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a clinical question."}), 400

    try:
        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an evidence-based clinical Q&A assistant. "
                        "Provide concise, accurate answers with guideline-level recommendations "
                        "and mention key trials or references when appropriate. "
                        "Do not fabricate evidence; say if something is uncertain."
                    ),
                },
                {"role": "user", "content": question},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        answer = ""
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {})
            answer = (msg.get("content") or "").strip()

        if not answer:
            answer = "No response generated."

        return jsonify({"answer": answer})

    except Exception as e:
        print("Error contacting model API:", e)
        return jsonify({"error": "There was an error generating the answer."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
