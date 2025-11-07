import os
import glob
import requests
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

if not API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY or OPENAI_API_KEY in .env")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # optional for web search

app = Flask(__name__, template_folder="templates", static_folder="static")

# -------- Optional: preload local guideline/protocol files from /data --------

DOCS = []
if os.path.isdir("data"):
    for path in glob.glob("data/*.txt"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                DOCS.append((os.path.basename(path), f.read()))
        except Exception as e:
            print(f"Error loading {path}: {e}")


def fetch_local_context(query: str, max_chars: int = 1800) -> str:
    """Very simple keyword-based retrieval over local txt docs."""
    if not DOCS:
        return ""
    q = query.lower()
    hits = []
    for name, content in DOCS:
        cl = content.lower()
        # crude relevance: any query word present
        if any(w and w in cl for w in q.split()):
            snippet = content[:max_chars]
            hits.append(f"From {name}:\n{snippet}")
    if not hits:
        return ""
    return "Local clinical documents:\n" + "\n\n".join(hits[:3])


# -------- PubMed retrieval --------

def fetch_pubmed_summaries(query: str, max_results: int = 5) -> str:
    """Fetch top PubMed records (titles, journals, years, PMIDs)."""
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

        summary_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        }
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
                pieces = [title]
                if journal:
                    pieces.append(f"{journal}")
                if year:
                    pieces.append(year)
                line = " - " + " | ".join(pieces) + f" | PMID:{pid}"
                lines.append(line)

        if not lines:
            return ""

        return "Relevant PubMed records:\n" + "\n".join(lines)
    except Exception as e:
        print("PubMed fetch error:", e)
        return ""


# -------- Optional: Serper-based medical web search --------

def search_web_medical(query: str, max_results: int = 5) -> str:
    """Use Serper.dev to pull reputable web sources. Safe if key missing."""
    if not SERPER_API_KEY:
        return ""
    try:
        payload = {
            "q": query
            + " site:who.int OR site:nice.org.uk OR site:ema.europa.eu OR site:nih.gov OR site:gov.au",
            "num": max_results,
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        r = requests.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=10)
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

    # Retrieve external context (lightweight RAG)
    pubmed_context = fetch_pubmed_summaries(question)
    web_context = search_web_medical(question)
    local_context = fetch_local_context(question)

    context_blocks = "\n\n".join(
        b for b in [pubmed_context, web_context, local_context] if b
    )

    # Build messages for the model
    messages = [
        {
            "role": "system",
            "content": (
                "You are an evidence-based clinical Q&A assistant for doctors.\n"
                "Use ONLY reliable medical sources (guidelines, major trials, PubMed, reputable agencies) "
                "including the provided context when available.\n"
                "Very important: Present the answer so that the most clinically useful information appears FIRST.\n"
                "Answer format (do NOT add headings unless they help clarity):\n"
                "1) Start with a concise, high-yield summary for clinicians (3–8 bullet points or short lines) "
                "covering diagnosis, initial management, red flags, and key numbers.\n"
                "2) Then provide a brief, structured explanation with rationale.\n"
                "3) Finish with a short list of key references or guideline names (e.g. 'NICE AF 2021', 'PMID 12345678').\n"
                "If evidence is uncertain or evolving, say so explicitly. Do not invent citations."
            ),
        },
    ]

    if context_blocks:
        messages.append(
            {
                "role": "system",
                "content": (
                    "External context (PubMed/web/local docs) for this query:\n"
                    f"{context_blocks}"
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.25,
        "max_tokens": 900,
    }

    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=70)
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
