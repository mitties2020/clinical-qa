import os
import json
import requests
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    Response,
    stream_with_context,
)
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY in .env")

app = Flask(__name__, template_folder="templates", static_folder="static")

# -------- Simple persistent cache --------
CACHE_PATH = "cache.json"

if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r") as f:
            cache = json.load(f)
    except Exception:
        cache = {}
else:
    cache = {}


def save_cache():
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print("Cache save error:", e)


# -------- Routes --------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate an answer using DeepSeek with streaming + caching."""
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "No query provided."}), 400

    key = query.lower()

    # ✅ Cache hit: return instantly as plain text
    if key in cache:
        return Response(cache[key], mimetype="text/plain")

    def stream():
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
                        "Use headings such as Summary, Assessment, Investigations, Treatment, "
                        "Monitoring, Follow-up & Safety Netting, Red Flags, and References "
                        "when relevant. Use professional wording and avoid markdown symbols like ** or bullet markers; "
                        "plain text lists are fine."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "temperature": 0.3,
            "max_tokens": 1600,
            "stream": True,
        }

        collected = []

        try:
            with requests.post(
                DEEPSEEK_URL,
                headers=headers,
                json=body,
                stream=True,
                timeout=70,
            ) as r:
                r.raise_for_status()

                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    # DeepSeek uses OpenAI-style streaming: "data: {...}"
                    if line.startswith("data: "):
                        chunk_str = line[len("data: "):].strip()
                    else:
                        chunk_str = line.strip()

                    if not chunk_str or chunk_str == "[DONE]":
                        if chunk_str == "[DONE]":
                            break
                        continue

                    try:
                        payload = json.loads(chunk_str)
                    except json.JSONDecodeError:
                        continue

                    choice = payload.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        collected.append(content)
                        # stream raw text out to the client
                        yield content

        except Exception as e:
            print("DeepSeek stream error:", e)
            yield f"\n[Error contacting model: {e}]"

        # After streaming, store full answer in cache
        full = "".join(collected).strip()
        if full:
            cache[key] = full
            save_cache()

    # Stream plain text back to the browser
    return Response(stream_with_context(stream()), mimetype="text/plain")


if __name__ == "__main__":
    # For local testing; in production Render/Gunicorn will run this differently
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
