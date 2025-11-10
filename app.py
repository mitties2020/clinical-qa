import os
import glob
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

# Core model config
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

if not API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY in .env")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# Optional: Serper for guideline search (if you have it)
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

app = Flask(__name__, template_folder="templates", static_folder="static")

# -------- Local documents (/data) --------

DOCS = []
if os.path.isdir("data"):
    for path in glob.glob("data/*.txt"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                DOCS.append((os.path.basename(path), f.read()))
        except Exception as e:
            print(f"Error loading {path}: {e}")


def fetch_local_context(query: str, max_chars: int = 2400) -> str:
    """Very simple keyword-based retrieval over local text docs."""
    if not DOCS:
        return ""
    q = query.lower()
    hits = []
    for name, content in DOCS:
        cl = content.lower()
        if any(w and w in cl for w in q.split()):
            snippet = content[:max_chars]
            hits.append(f"From {name}:\n{snippet}")
    if not hits:
        return ""
    return "Local clinical documents:\n" + "\n\n".join(hits[:3])


# -------- PubMed (evidence context) --------

def fetch_pubmed_summaries(query: str, max_results: int = 5) -> str:
    """Fetch concise PubMed references for context."""
    try:
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance",
        }
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return ""

        summary_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        s = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params=summary_params,
            timeout=10,
        )
        s.raise_for_status()
        data = s.json().get("result", {})

        lines = []
        for pid in ids:
            item = data.get(pid, {})
            title = item.get("title")
            journal = item.get("fulljournalname") or item.get("source")
            year = (item.get("pubdate") or "").split(" ")[0]
            if title:
                parts = [title]
                if journal:
                    parts.append(journal)
                if year:
                    parts.append(year)
                lines.append(" - " + " | ".join(parts) + f" | PMID:{pid}")

        if not lines:
            return ""
        return "Relevant PubMed records:\n" + "\n".join(lines)

    except Exception as e:
        print("PubMed fetch error:", e)
        return ""


# -------- Optional: guideline web search via Serper --------

def search_web_medical(query: str, max_results: int = 5) -> str:
    """Search selected trusted domains if SERPER_API_KEY is set."""
    if not SERPER_API_KEY:
        return ""
    try:
        payload = {
            "q": (
                query
                + " site:who.int OR site:nice.org.uk OR site:ema.europa.eu "
                  "OR site:nih.gov OR site:gov.au OR site:idsociety.org"
            ),
            "num": max_results,
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        r = requests.post(
            "https://google.serper.dev/search",
            json=payload,
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        items = (data.get("organic") or [])[:max_results]
        lines = []
        for item in items:
            title = item.get("title")
            url = item.get("link")
            if title and url:
                lines.append(f"- {title} ({url})")
        if not lines:
            return ""
        return "Relevant web sources:\n" + "\n".join(lines)
    except Exception as e:
        print("Web search error:", e)
        return ""


# -------- Routes --------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}
    question = (data.get("query") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a clinical question."}), 400

    # External context: local docs + PubMed + optional web
    local_context = fetch_local_context(question)
    pubmed_context = fetch_pubmed_summaries(question)
    web_context = search_web_medical(question)

    context_blocks = "\n\n".join(
        b for b in [local_context, pubmed_context, web_context] if b
    )

    # System message enforces depth + structure for collapsible UI
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-based clinical Q&A assistant for doctors in acute and outpatient care.\n"
                "You MUST answer in a structured, hierarchical format using the following sections "
                "in this exact order where relevant:\n"
                "Summary:\n"
                "Assessment:\n"
                "Investigations:\n"
                "Treatment:\n"
                "Monitoring:\n"
                "Follow-up & Safety Netting:\n"
                "Red Flags:\n"
                "References:\n\n"
                "Formatting rules (very important):\n"
                "- Do NOT use Markdown bold (**like this**) or decorative asterisks.\n"
                "- Each section name MUST end with a colon on its own line (e.g. 'Treatment:').\n"
                "- Under each section, use short bullet or numbered lines with rich clinical detail "
                "(doses, durations, thresholds, criteria) suitable for registrars/consultants.\n"
                "- In the Treatment section, where specific options are recommended, use the pattern:\n"
                "  'Option: <name>' on one line, followed by one or more lines starting with 'Detail:' "
                "to explain indications, dosing, adjustments, cautions.\n"
                "  Example:\n"
                "    Option: Apixaban\n"
                "    Detail: Standard dose 5 mg BD; reduce to 2.5 mg BD if ...\n"
                "- References section: include 3–8 key items (guidelines, societies, PMIDs) in plain text.\n"
                "- Be comprehensive but clinically concise. Do not invent citations or data.\n"
                "- If uncertain or evidence-limited, explicitly state the uncertainty.\n"
            ),
        },
    ]

    if context_blocks:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Use the following external context (local documents, PubMed, guidelines) to inform your answer:\n"
                    f"{context_blocks}"
                ),
            }
        )

    messages.append({"role": "user", "content": question})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.22,
        "max_tokens": 1400,  # allow deeper detail
    }

    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=80)
        resp.raise_for_status()
        res_data = resp.json()

        answer = ""
        if "choices" in res_data and res_data["choices"]:
            msg = res_data["choices"][0].get("message", {})
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
