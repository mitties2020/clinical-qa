import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY in environment. "
                       "Set DEEPSEEK_API_KEY in a .env file or your hosting env.")

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "No query provided."}), 400

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        body = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a clinical decision support assistant for doctors. "
                        "Provide concise, high-yield, evidence-based answers with clear structure. "
                        "Where relevant, organise under headings such as Summary, Assessment, "
                        "Investigations, Treatment, Monitoring, Follow-up & Safety Netting, "
                        "Red Flags, and References. "
                        "Use professional wording. Avoid markdown syntax like ** or bullet symbols; "
                        "write plain text lists and clear paragraphs so the frontend can format."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "temperature": 0.3,
            "max_tokens": 1600,
            "stream": False,
        }

        resp = requests.post(DEEPSEEK_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Extract answer text safely
        answer = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            return jsonify({"error": "No response from model."}), 500

        return jsonify({"answer": answer})

    except requests.exceptions.RequestException as e:
        print("DeepSeek API error:", e)
        return jsonify({"error": "Error contacting model API."}), 500
    except Exception as e:
        print("Server error:", e)
        return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    # Local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
