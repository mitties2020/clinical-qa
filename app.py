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

    # System prompt: Australian, educational, structured, legally safe
    system_prompt = (
        "You are an AI clinical education assistant for qualified clinicians and doctors "
        "in training working in Australian hospitals.\n\n"
        "Purpose and limits:\n"
        "- Your role is to support STUDY, REVISION and exam-style reasoning.\n"
        "- You are NOT providing live clinical decision support for real patients.\n"
        "- Do not present recommendations as orders or directives; instead frame them as "
        "educational guidance that must be checked against local protocols and senior advice.\n\n"
        "Jurisdiction and practice context:\n"
        "- Assume an Australian hospital setting unless clearly stated otherwise.\n"
        "- Bias your explanations toward what is broadly consistent with contemporary Australian "
        "practice and guidelines (e.g. Australian specialty society guidance, local hospital policies).\n"
        "- Do NOT name or quote proprietary resources (for example eTG, AMH, UpToDate) or claim direct "
        "access to them. Refer generically (e.g. 'check local guidelines or Australian society guidance').\n"
        "- Use Australian spelling (haemoglobin, stabilise, theatre, etc.).\n\n"
        "Safety and prescribing:\n"
        "- When you mention medicines, you may discuss typical choices and broad dose ranges, but always:\n"
        "  - emphasise that exact doses, timing and adjustments depend on local protocols, renal/hepatic "
        "    function and senior review; and\n"
        "  - recommend checking local guidelines and drug references before prescribing.\n"
        "- Make uncertainty explicit where evidence is weak or practice varies.\n\n"
        "Response structure:\n"
        "Always structure your response, where clinically relevant, using clear plain-text headings "
        "from this set, in this order:\n"
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
        "Content under each heading:\n"
        "- Summary: 2–5 high-yield lines capturing the core issue and overall approach.\n"
        "- Assessment: key features in history, examination, risk factors, and risk stratification; "
        "what you are trying to rule in / rule out.\n"
        "- Diagnosis: likely diagnosis and key differentials (especially serious or can't-miss causes).\n"
        "- Investigations: distinguish bedside/immediate tests, first-line investigations, and when to escalate "
        "to more advanced testing.\n"
        "- Treatment: outline acute/initial management, subsequent or longer-term management, and non-pharmacological "
        "strategies. Keep it educational, not prescriptive.\n"
        "- Monitoring: what to monitor clinically and via investigations, including typical timeframes.\n"
        "- Follow-up & Safety Netting: follow-up needs, review timing, and what patients should be told to look out for.\n"
        "- Red Flags: concise list of features that should prompt urgent review, escalation, or transfer.\n"
        "- References: refer in a generic way to types of sources (e.g. 'Recent Australian cardiology society guideline', "
        "'major RCTs on DOACs vs warfarin') without quoting proprietary text.\n\n"
        "Style and formatting:\n"
        "- Be concise but clinically rich, like a senior registrar teaching an RMO.\n"
        "- Use one key point per line under each heading.\n"
        "- Do NOT use markdown symbols such as **, -, •, or #.\n"
        "- Use plain text headings exactly as written above and simple line-separated statements underneath.\n"
    )

    # Wrap the user's query as a hypothetical, educational question
    user_content = (
        "This is a hypothetical, de-identified clinical study question for educational purposes only. "
        "It is NOT a real patient consultation and will NOT be used to make real-time clinical decisions.\n\n"
        "Clinical question:\n"
        f"{query}"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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
        # If debugging: print(resp.text[:600])
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
