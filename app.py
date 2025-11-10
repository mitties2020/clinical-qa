import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY. Set it in your environment or .env file.")

session = requests.Session()
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "No query provided."}), 400

    system_prompt = (
        "You are a clinical decision support assistant for qualified clinicians.\n"
        "Provide concise but high-yield, evidence-based answers.\n"
        "Always structure your response where relevant using clear headings in this order:\n"
        "Summary\n"
        "Assessment\n"
        "Investigations\n"
        "Treatment\n"
        "Monitoring\n"
        "Follow-up & Safety Netting\n"
        "Red Flags\n"
        "References\n"
        "Only include headings that are clinically relevant for the query.\n"
        "Under each heading, use short, direct points (one per line), prioritising the most important actions first.\n"
        "Include practical details (drug choices, doses, thresholds, timeframes) when appropriate, "
        "but avoid long narrative paragraphs.\n"
        "DO NOT use markdown formatting symbols like **, bullets, or #; just plain text headings and lines. "
        "The UI will format it."
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0.25,
        "top_p": 0.9,
        # Slightly higher for more depth but still safe for speed
        "max_tokens": 1100,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        resp = session.post(DEEPSEEK_URL, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        answer = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not answer:
            return jsonify({"error": "No response from model."}), 502

        return jsonify({"answer": answer})

    except requests.exceptions.Timeout:
        return jsonify({"error": "Upstream model timed out. Please try again."}), 504
    except requests.exceptions.RequestException as e:
        print("DeepSeek API error:", repr(e))
        return jsonify({"error": "Error contacting model API."}), 502
    except Exception as e:
        print("Server error:", repr(e))
        return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
