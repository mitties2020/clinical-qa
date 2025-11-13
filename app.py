import os
import requests
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    abort,
)
from dotenv import load_dotenv

# =====================================
# Environment / DeepSeek configuration
# =====================================

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

if not DEEPSEEK_API_KEY:
    raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable.")


# =====================================
# Flask app
# =====================================

app = Flask(__name__, template_folder="templates", static_folder="static")


# =====================================
# Request logging (simple traffic monitor)
# =====================================

@app.before_request
def log_request_info():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    ua = request.headers.get("User-Agent", "unknown")
    path = request.path
    method = request.method
    print(f"[TRACK] {ip} {method} {path} {ua}")


# =====================================
# Clinical calculators configuration
# (no paediatric medication dosing)
# =====================================

CALCULATORS = [
    {
        "slug": "heart",
        "name": "HEART Score",
        "category": "Cardiology / ED",
        "description": "History, ECG, Age, Risk factors, Troponin chest pain risk score."
    },
    {
        "slug": "timi-ua-nstemi",
        "name": "TIMI Risk Score (UA/NSTEMI)",
        "category": "Cardiology / ED",
        "description": "Seven-item risk score for UA/NSTEMI patients."
    },
    {
        "slug": "wells-pe",
        "name": "Wells Score – Pulmonary Embolism",
        "category": "Respiratory / VTE",
        "description": "Pre-test probability of pulmonary embolism."
    },
    {
        "slug": "perc",
        "name": "PERC Rule",
        "category": "Respiratory / VTE",
        "description": "Pulmonary Embolism Rule-out Criteria for low-risk patients."
    },
    {
        "slug": "wells-dvt",
        "name": "Wells Score – DVT",
        "category": "VTE",
        "description": "Pre-test probability of lower limb DVT."
    },
    {
        "slug": "chadsvasc",
        "name": "CHA\u2082DS\u2082-VASc",
        "category": "Cardiology",
        "description": "Stroke risk stratification in non-valvular atrial fibrillation."
    },
    {
        "slug": "has-bled",
        "name": "HAS-BLED",
        "category": "Cardiology / Haematology",
        "description": "Bleeding risk score for patients on anticoagulation."
    },
    {
        "slug": "curb65",
        "name": "CURB-65",
        "category": "Respiratory / Infectious Diseases",
        "description": "Community-acquired pneumonia severity score."
    },
    {
        "slug": "qsofa",
        "name": "qSOFA",
        "category": "Sepsis",
        "description": "Quick sepsis-related organ failure assessment."
    },
]


@app.context_processor
def inject_calculators():
    """
    Make CALCULATORS available to all templates (for navbar dropdown etc.)
    """
    return {"CALCULATORS": CALCULATORS}


# =====================================
# DeepSeek helper
# =====================================

def call_deepseek(messages):
    """
    Call DeepSeek chat completion API with a list of messages.
    messages = [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      ...
    ]
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Standard OpenAI/DeepSeek-style response
    return data["choices"][0]["message"]["content"]


# =====================================
# Core routes (homepage + Q&A API)
# =====================================

@app.route("/")
def index():
    """
    Render main page with your question form / UI.
    """
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """
    Endpoint your frontend calls to get an answer from DeepSeek.

    Expects JSON: { "question": "..." }
    Returns JSON: { "answer": "..." }
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400

    # You can tune the system prompt however you like
    system_prompt = (
        "You are an Australian clinical education assistant. "
        "You provide evidence-based, guideline-aware information using Australian spelling, "
        "but you do not give individual treatment instructions or dosing. "
        "You always remind users that your answers are general educational information only "
        "and not a substitute for local guidelines or senior clinician review."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        answer = call_deepseek(messages)
    except Exception as e:
        print(f"[ERROR] DeepSeek API failed: {e}")
        return jsonify({"error": "Backend error contacting model."}), 500

    return jsonify({"answer": answer})


@app.route("/ping")
def ping():
    """
    Simple health check.
    """
    return "ok", 200


# =====================================
# Calculator routes
# =====================================

@app.route("/calculators")
def calculators_index():
    """
    List all calculators in a grid.
    Template: templates/calculators_index.html
    """
    return render_template("calculators_index.html", calculators=CALCULATORS)


@app.route("/calculators/<slug>", methods=["GET", "POST"])
def calculator_view(slug):
    """
    Generic calculator dispatcher.
    Each slug maps to a template and scoring logic.
    """
    calc = next((c for c in CALCULATORS if c["slug"] == slug), None)
    if not calc:
        abort(404)

    context = {"calc": calc}
    template = "calculators/coming_soon.html"  # fallback if not implemented

    # -------------------
    # HEART SCORE
    # -------------------
    if slug == "heart":
        template = "calculators/heart.html"
        if request.method == "POST":
            try:
                history = int(request.form.get("history", 0))
                ecg = int(request.form.get("ecg", 0))
                age = int(request.form.get("age", 0))
                risk = int(request.form.get("risk", 0))
                troponin = int(request.form.get("troponin", 0))
                score = history + ecg + age + risk + troponin
                context["score"] = score
            except ValueError:
                context["error"] = "Please select values for all fields."

    # -------------------
    # TIMI UA/NSTEMI
    # -------------------
    elif slug == "timi-ua-nstemi":
        template = "calculators/timi_ua_nstemi.html"
        if request.method == "POST":
            items = [
                "age65",
                "three_risks",
                "known_cad",
                "asa_7d",
                "severe_angina",
                "st_deviation",
                "elevated_markers",
            ]
            score = 0
            for key in items:
                if request.form.get(key):
                    score += 1
            context["score"] = score

    # -------------------
    # WELLS – PE
    # -------------------
    elif slug == "wells-pe":
        template = "calculators/wells_pe.html"

        if request.method == "POST":
            score = 0.0

            def add_if(name, points):
                nonlocal score
                if request.form.get(name):
                    score += points

            add_if("dvt_signs", 3.0)
            add_if("pe_more_likely", 3.0)
            add_if("hr_gt_100", 1.5)
            add_if("immobilisation", 1.5)
            add_if("prev_vte", 1.5)
            add_if("haemoptysis", 1.0)
            add_if("malignancy", 1.0)

            context["score"] = score

    # -------------------
    # PERC RULE
    # -------------------
    elif slug == "perc":
        template = "calculators/perc.html"

        if request.method == "POST":
            criteria = [
                "age_ge_50",
                "hr_ge_100",
                "sat_lt_95",
                "unilateral_swelling",
                "haemoptysis",
                "recent_surgery",
                "prior_vte",
                "oestrogen",
            ]
            positive = 0
            for key in criteria:
                if request.form.get(key):
                    positive += 1
            context["positive"] = positive
            context["perc_negative"] = (positive == 0)

    # -------------------
    # WELLS – DVT
    # -------------------
    elif slug == "wells-dvt":
        template = "calculators/wells_dvt.html"

        if request.method == "POST":
            score = 0

            def add_if(name, points):
                nonlocal score
                if request.form.get(name):
                    score += points

            add_if("active_cancer", 1)
            add_if("paralysis", 1)
            add_if("bedridden", 1)
            add_if("tenderness", 1)
            add_if("entire_leg_swollen", 1)
            add_if("calf_gt_3cm", 1)
            add_if("pitting_oedema", 1)
            add_if("collateral_veins", 1)
            add_if("prev_dvt", 1)
            if request.form.get("alt_dx_more_likely"):
                score -= 2

            context["score"] = score

    # -------------------
    # CHA2DS2-VASc
    # -------------------
    elif slug == "chadsvasc":
        template = "calculators/chadsvasc.html"

        if request.method == "POST":
            score = 0

            def add_if(name, points):
                nonlocal score
                if request.form.get(name):
                    score += points

            add_if("chf", 1)
            add_if("htn", 1)
            add_if("age_75", 2)
            add_if("diabetes", 1)
            add_if("stroke_tia", 2)
            add_if("vascular", 1)
            add_if("age_65_74", 1)
            add_if("female", 1)

            context["score"] = score

    # -------------------
    # HAS-BLED
    # -------------------
    elif slug == "has-bled":
        template = "calculators/has_bled.html"

        if request.method == "POST":
            score = 0

            def add_if(name, points):
                nonlocal score
                if request.form.get(name):
                    score += points

            add_if("htn", 1)
            add_if("renal", 1)
            add_if("liver", 1)
            add_if("stroke", 1)
            add_if("bleeding", 1)
            add_if("labile_inr", 1)
            add_if("elderly_65", 1)
            add_if("drugs", 1)
            add_if("alcohol", 1)

            context["score"] = score

    # -------------------
    # CURB-65
    # -------------------
    elif slug == "curb65":
        template = "calculators/curb65.html"

        if request.method == "POST":
            score = 0
            criteria = ["confusion", "urea", "rr_30", "bp_low", "age_65"]
            for key in criteria:
                if request.form.get(key):
                    score += 1
            context["score"] = score

    # -------------------
    # qSOFA
    # -------------------
    elif slug == "qsofa":
        template = "calculators/qsofa.html"

        if request.method == "POST":
            score = 0
            if request.form.get("sbp_le_100"):
                score += 1
            if request.form.get("rr_ge_22"):
                score += 1
            if request.form.get("altered_mentation"):
                score += 1
            context["score"] = score

    return render_template(template, **context)


# =====================================
# Main entrypoint
# =====================================

if __name__ == "__main__":
    # For local dev; in production use gunicorn or similar
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
