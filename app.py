import os
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

# ===== CONFIG =====
API_KEY = os.getenv("DEEPSEEK_API_KEY")  # put your key in .env or hardcode if needed
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not API_KEY:
    raise RuntimeError(
        "Missing DEEPSEEK_API_KEY. Set it in a .env file or your environment."
    )

# Reuse one HTTP session for all requests (reduces SSL/TCP overhead)
session = requests.Session()

app = Flask(__name__, template_folder="templates", static_folder="static")


# ===== ROUTES =====

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "No query provided."}), 400

    # Lean, fast system prompt: keeps answers clinical and structured, but not bloated.
    system_prompt = (
        "You are a clinical decision support assistant for doctors. "
        "Provide concise, high-yield, evidence-based answers only. "
        "Prioritise the most important information first for front-line decision-making. "
        "Where appropriate, structure content using clear headings such as: "
        "Summary, Assessment, Investigations, Treatment, Monitoring, Follow-up & Safety Netting, "
        "Red Flags, and References. "
        "Avoid markdown symbols (no **, no bullet characters). "
        "Use short, direct sentences and plain text lists so the UI can format bullets. "
        "Do not generate long essays or repeated explanations."
    )

    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        # Slightly lower temperature for consistency and speed
        "temperature": 0.2,
        "top_p": 0.9,
        # Lower max_tokens = faster responses but still detailed enough
        "max_tokens": 800,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        resp = session.post(
            DEEPSEEK_URL,
            headers=headers,
            json=body,
            timeout=25,  # fail fast if model is hanging
        )
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
        # If DeepSeek is too slow, return a clear message instead of hanging forever
        return jsonify({"error": "Upstream model timed out. Please try again."}), 504

    except requests.exceptions.RequestException as e:
        print("DeepSeek API error:", repr(e))
        return jsonify({"error": "Error contacting model API."}), 502

    except Exception as e:
        print("Server error:", repr(e))
        return jsonify({"error": "Internal server error."}), 500


# ===== ENTRYPOINT =====

if __name__ == "__main__":
    # debug=False to avoid double-loading & extra overhead
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
