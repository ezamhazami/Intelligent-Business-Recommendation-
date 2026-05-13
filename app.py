import json
import os
import pickle
import re
from google import genai
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()
app = Flask(__name__)

MODEL_PATH   = "models/rf_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
EVAL_PATH    = "models/evaluation_report.json"
FEATURE_PATH = "models/feature_names.pkl"
DATASET_PATH = "data/shah_alam_labeled2.csv"

model, label_encoder, feature_names, eval_report, dataset = None, None, None, {}, None


def load_artefacts():
    global model, label_encoder, feature_names, eval_report, dataset
    try:
        with open(MODEL_PATH,   "rb") as f: model         = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f: label_encoder = pickle.load(f)
        with open(FEATURE_PATH, "rb") as f: feature_names = pickle.load(f)
        with open(EVAL_PATH,    "r")  as f: eval_report   = json.load(f)
        dataset = pd.read_csv(DATASET_PATH)
        print("ML Artefacts Loaded Successfully")
    except Exception as e:
        print(f"Could not load artefacts: {e}")


load_artefacts()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "")
client         = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
print("KEY LOADED:", bool(GOOGLE_API_KEY))

# ── Feature descriptions ──────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "population":              "Number of population",
    "food_beverage":           "Food & Beverage outlet density",
    "retail_outlet":           "Retail outlet density",
    "service_business":        "Service business density (salons, clinics, repair shops)",
    "entertainment":           "Entertainment facility density (gyms, cinemas)",
    "educational_inst":        "Educational institution density (schools, universities)",
    "corporate_office":        "Corporate office density (professional workforce)",
    "financial_inst":          "Financial institution density (economic activity indicator)",
    "shopping_mall":           "Shopping mall density (high consumer traffic)",
    "automotive":              "Automotive workshop density (workshop)",
    "healthcare":              "Healthcare facility density (hospitals, clinics, pharmacies)",
    "transportation":          "Transportation accessibility (public transport, roads, connectivity)",
    "amenity_diversity_index": "Overall amenity diversity of the section",
}

EXPLANATION_KEYS = [
    "population", "food_beverage", "retail_outlet", "service_business",
    "entertainment", "educational_inst", "corporate_office", "financial_inst",
    "shopping_mall", "automotive", "healthcare", "transportation",
    "amenity_diversity_index",
]

# ── Business insights ─────────────────────────────────────────────────────────
BUSINESS_INSIGHTS = {
    "Food & Beverage":    "This section has a strong customer base and established food demand.",
    "Retail & Commerce":  "Mall presence and retail clustering support strong consumer footfall.",
    "Community Services": "Population demand and service gaps indicate opportunity for essential services.",
    "Business & Trade":   "Corporate, automotive, and financial signals indicate a trade-oriented area.",
    "Leisure & Lifestyle":"Entertainment and connectivity support lifestyle-oriented businesses.",
}

# ── Intent patterns ───────────────────────────────────────────────────────────
RECOMMENDATION_KEYWORDS = [
    "recommend", "business", "open", "start", "suggest",
    "best", "suitable", "what business", "which business",
    "where should", "location", "area", "place",
]
GREETING_PATTERNS = [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bsalam\b"]
HELP_PATTERNS     = [r"\bhelp\b", r"\bwhat can\b", r"\bwhat do\b"]
STATS_PATTERNS    = [
    r"\baccuracy\b", r"\bmodel\b", r"\bperformance\b",
    r"\bprecision\b", r"\brecall\b", r"\bf1\b",
    r"\bstats\b",    r"\bstatistics\b",
]
SECTION_LIST_PATS = [
    r"\blist\b",
    r"\ball sections?\b",
    r"\bshow sections?\b",
    r"\bwhat sections?\b",
    r"\bavailable sections?\b",
    r"\ball area\b",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _section_sort_key(s: str):
    part = s.split()[-1]
    try:
        return (0, int(part), "")
    except ValueError:
        return (1, 0, part)


# ── Budget extraction ─────────────────────────────────────────────────────────
def extract_budget(text: str) -> tuple[int | None, str]:
    t     = text.lower().replace(",", "")
    match = re.search(r"(?:rm|budget\s*(?:of|is)?)\s*([\d]+\.?\d*)\s*(k?)", t)
    if match:
        amount = float(match.group(1))
        if match.group(2) == "k":
            amount *= 1000
        amount = int(amount)

        if amount <= 5000:
            tier = (
                f"The entrepreneur has RM {amount:,} to invest (micro budget). "
                "Recommend ONLY home-based, online, or mobile concepts — no physical shopfront. "
                "Examples: food delivery ghost kitchen from home using GrabFood/Foodpanda, "
                "online dropshipping store, mobile service (grooming, repair, tutoring). "
                "Every idea in Paragraph 3 MUST cost under RM 5,000 to start."
            )
        elif amount <= 20000:
            tier = (
                f"The entrepreneur has RM {amount:,} to invest (small budget). "
                "Recommend small stall, kiosk, or shared-space concepts. "
                "Examples: food stall in pasar malam or food court, small booth in a mall, "
                "home-based catering, tuition from a rented room, small online retail. "
                "Every idea in Paragraph 3 MUST be achievable under RM 20,000."
            )
        elif amount <= 50000:
            tier = (
                f"The entrepreneur has RM {amount:,} to invest (medium budget). "
                "A small shophouse unit or modest standalone outlet is viable. "
                "Examples: F&B outlet, small retail shop, salon, convenience store, "
                "budget franchise. Every idea in Paragraph 3 MUST fit within RM 50,000."
            )
        else:
            tier = (
                f"The entrepreneur has RM {amount:,} to invest (substantial budget). "
                "Full shophouse, established franchise, or multi-staff operation is viable. "
                "Examples: restaurant, retail chain outlet, corporate services firm, gym, "
                "established franchise brand. Recommend ambitious but realistic concepts."
            )
        return amount, tier

    return None, (
        "No budget was specified. In Paragraph 3, recommend one low-cost option "
        "(under RM 20,000, e.g. stall or online) AND one higher-investment option "
        "(RM 50,000+, e.g. shophouse or franchise), so the entrepreneur can self-select."
    )


# ── User business extraction ──────────────────────────────────────────────────
def extract_user_business(text: str) -> str | None:
    patterns = [
        r"can i open (?:a |an )?([\w\s&]+?) (?:business|shop|store|restaurant|cafe|outlet) in",
        r"is (?:a |an )?([\w\s&]+?) (?:business|shop|store|restaurant|cafe|outlet) good for",
        r"(?:open|start|run) (?:a |an )?([\w\s&]+?) (?:business|shop|store|restaurant|cafe|outlet) in section",
        r"(?:what about|how about) (?:a |an )?([\w\s&]+?) (?:business|shop|store|restaurant|cafe|outlet)",
        r"(?:i want to open|i'm thinking of opening|thinking of) (?:a |an )?([\w\s&]+?) in section",
        r"(?:suitable|good) (?:for|to open) (?:a |an )?([\w\s&]+?) in section",
        r"(?:evaluate|assess) (?:a |an )?([\w\s&]+?) (?:business|shop|store)? in section",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


# ── Intent detection ──────────────────────────────────────────────────────────
def detect_intent(text: str) -> str:
    t = text.lower()
    if any(re.search(p, t) for p in GREETING_PATTERNS): return "greeting"
    if any(re.search(p, t) for p in HELP_PATTERNS):     return "help"
    if any(re.search(p, t) for p in STATS_PATTERNS):    return "stats"
    # list_sections BEFORE recommend
    if any(re.search(p, t) for p in SECTION_LIST_PATS): return "list_sections"
    # evaluate if user proposes their own business + section number
    if re.search(r"section\s*\d+", t) and extract_user_business(t): return "evaluate"
    # recommend only when there's an actual section NUMBER
    if re.search(r"section\s*\d+", t):                  return "recommend"
    if any(kw in t for kw in RECOMMENDATION_KEYWORDS):  return "recommend"
    return "general"


def extract_section(text: str) -> str | None:
    match = re.search(r"section\s*(\d+)", text, re.IGNORECASE)
    if match:
        return f"Section {match.group(1)}"
    return None


# ── Feature importance ────────────────────────────────────────────────────────
def _top3_business_features() -> list[tuple[str, float]]:
    rf      = eval_report.get("random_forest", eval_report) if isinstance(eval_report, dict) else {}
    raw_imp = rf.get("feature_importance", {}) if isinstance(rf, dict) else {}
    agg_imp = {k: 0.0 for k in EXPLANATION_KEYS}

    if isinstance(raw_imp, dict) and raw_imp:
        for feat_name, score in raw_imp.items():
            for base in EXPLANATION_KEYS:
                if feat_name == base or feat_name.startswith(base + "_"):
                    agg_imp[base] += float(score)
                    break
    else:
        rf_step = model.named_steps["model"]
        for f, i in zip(feature_names, rf_step.feature_importances_):
            if f in agg_imp:
                agg_imp[f] += float(i)

    feat_imp = sorted(agg_imp.items(), key=lambda x: x[1], reverse=True)
    return [(f, round(float(i), 4)) for f, i in feat_imp[:3]]


# ── ML Prediction ─────────────────────────────────────────────────────────────
def predict_for_section(section_name: str) -> dict | None:
    if dataset is None or model is None:
        return None

    row_df = dataset[dataset["section"].str.lower() == section_name.lower()]
    if row_df.empty:
        return None

    row          = row_df.iloc[0]
    X            = pd.DataFrame([[row[f] for f in feature_names]], columns=feature_names)
    pred_encoded = model.predict(X)[0]
    pred_label   = label_encoder.inverse_transform([pred_encoded])[0]
    probas       = model.predict_proba(X)[0]
    class_labels = label_encoder.inverse_transform(range(len(probas)))

    top3_idx    = np.argsort(probas)[::-1][:3]
    top3_labels = [(class_labels[i], round(float(probas[i]) * 100, 1)) for i in top3_idx]
    top3_feats  = _top3_business_features()

    def _fmt(col, value):
        if col == "amenity_diversity_index":
            return round(float(value), 4)
        try:
            n = float(value)
            return int(n) if n.is_integer() else round(n, 4)
        except (TypeError, ValueError):
            return str(value)

    section_data = {f: _fmt(f, row[f]) for f in EXPLANATION_KEYS if f in row.index}

    return {
        "section":          section_name,
        "predicted_type":   pred_label,
        "confidence":       round(float(max(probas)) * 100, 1),
        "top3_predictions": top3_labels,
        "top3_features":    top3_feats,
        "section_data":     section_data,
    }


# ── NLG — standard recommendation ────────────────────────────────────────────
def generate_nlg_response(prediction: dict, user_message: str = "") -> str:
    section = prediction["section"]
    btype   = prediction["predicted_type"]
    top3_f  = prediction["top3_features"]
    s_data  = prediction["section_data"]

    f1_name, f1_val = top3_f[0][0], top3_f[0][1]
    f2_name, f2_val = top3_f[1][0], top3_f[1][1]
    f3_name, f3_val = top3_f[2][0], top3_f[2][1]

    f1_desc = FEATURE_DESCRIPTIONS.get(f1_name, f1_name)
    f2_desc = FEATURE_DESCRIPTIONS.get(f2_name, f2_name)
    f3_desc = FEATURE_DESCRIPTIONS.get(f3_name, f3_name)

    business_context = BUSINESS_INSIGHTS.get(btype, "")

    section_profile = "\n".join([
        f"  - {FEATURE_DESCRIPTIONS.get(k, k)}: {v}"
        for k, v in s_data.items()
    ])

    budget_amount, budget_instruction = extract_budget(user_message)
    budget_block = (
        f"\n=== BUDGET CONSTRAINT (MANDATORY) ===\n"
        f"{budget_instruction}\n"
        f"Violating the budget constraint means giving harmful advice — do not do it.\n"
    )

    if client:
        prompt = f"""You are a sharp, specific business advisor for Shah Alam, Malaysia.
{budget_block}
=== {section} DATA ===
{section_profile}

=== RECOMMENDATION ===
Business type: {btype}

Top 3 signals:
1. {f1_desc}: {s_data.get(f1_name)} (importance: {f1_val:.1%})
2. {f2_desc}: {s_data.get(f2_name)} (importance: {f2_val:.1%})
3. {f3_desc}: {s_data.get(f3_name)} (importance: {f3_val:.1%})

=== DOMAIN KNOWLEDGE ===
{business_context}

=== OUTPUT FORMAT (follow exactly) ===

Paragraph 1 — THE RECOMMENDATION (2 sentences):
- Start with: "For {section}, I recommend opening a **[specific business name]** under the {btype} category."
- Then explain the single strongest reason WHY using the top feature count. Use the actual number — do not use vague phrases like "strong presence" or "ripe for".

Paragraph 2 — WHY THIS SECTION (2-3 sentences):
- Explain what makes {section} specifically suitable using ALL 3 top feature counts.
- Use the actual numbers. Do not be vague.

Paragraph 3 — SPECIFIC BUSINESS IDEAS (3-4 sentences):
- Start DIRECTLY with the first business idea — no opening summary sentence.
- Recommend 2-3 SPECIFIC business concepts that strictly fit the budget constraint above.
- For each, name the business type AND directly link it to a specific count from the data.
- If budget is specified, ALL ideas must fit within it — no exceptions.
- Each idea must have a DIFFERENT name from the headline recommendation in Paragraph 1.
- Mention the specific budget amount when introducing the ideas.

Paragraph 4 — WATCH-OUT (1-2 sentences):
- Identify one risk or timing consideration tied to a SPECIFIC number in the data.

STRICT RULES:
- Bold every specific business recommendation using **bold**
- Use actual numbers in every paragraph — no vague statements
- Never mention ML, machine learning, algorithms, or confidence scores
- Do NOT say "shopping malls nearby" if the count refers to density within the section
- Do NOT add a summary sentence at the end of Paragraph 3 — end with the last specific business idea
- Do NOT capitalise feature names mid-sentence (write "corporate office density" not "Corporate office density")
- Do NOT use marketing phrases like "ripe for", "significant segment", or "strong presence" — use numbers instead
- Write with confidence — do not hedge or defend the recommendation
- Write in second person, flowing prose — no bullet points, no headers"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini ERROR] {e}")

    # ── Template fallback ─────────────────────────────────────────────────────
    high_features = sorted(
        s_data.items(),
        key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
        reverse=True,
    )
    top_feat_name, top_feat_val = high_features[0]
    top_feat_desc = FEATURE_DESCRIPTIONS.get(top_feat_name, top_feat_name)

    return (
        f"Based on geospatial analysis of **{section}**, I recommend a **{btype}** business.\n\n"
        f"{section} has notably high {top_feat_desc} ({top_feat_val}), "
        f"alongside {f1_desc} ({s_data.get(f1_name)}), "
        f"{f2_desc} ({s_data.get(f2_name)}), and "
        f"{f3_desc} ({s_data.get(f3_name)}) — "
        f"all pointing to strong **{btype}** potential.\n\n"
        f"{business_context}"
    )


# ── NLG — user business evaluation ───────────────────────────────────────────
def generate_evaluation_response(prediction: dict, user_business: str, user_message: str = "") -> str:
    section = prediction["section"]
    btype   = prediction["predicted_type"]
    top3_f  = prediction["top3_features"]
    s_data  = prediction["section_data"]

    f1_name, f1_val = top3_f[0][0], top3_f[0][1]
    f2_name, f2_val = top3_f[1][0], top3_f[1][1]
    f3_name, f3_val = top3_f[2][0], top3_f[2][1]

    f1_desc = FEATURE_DESCRIPTIONS.get(f1_name, f1_name)
    f2_desc = FEATURE_DESCRIPTIONS.get(f2_name, f2_name)
    f3_desc = FEATURE_DESCRIPTIONS.get(f3_name, f3_name)

    section_profile = "\n".join([
        f"  - {FEATURE_DESCRIPTIONS.get(k, k)}: {v}"
        for k, v in s_data.items()
    ])

    budget_amount, budget_instruction = extract_budget(user_message)
    budget_block = (
        f"\n=== BUDGET CONSTRAINT ===\n{budget_instruction}\n"
        if budget_amount else ""
    )

    if client:
        prompt = f"""You are a sharp, specific business advisor for Shah Alam, Malaysia.
An entrepreneur wants to open a **{user_business}** business in {section}, Shah Alam.
Your job is to evaluate whether this is a good or bad idea based on the actual section data.
{budget_block}
=== {section} DATA ===
{section_profile}

=== WHAT THE DATA RECOMMENDS FOR THIS SECTION ===
The data most strongly supports: {btype}
Top 3 signals for this section:
1. {f1_desc}: {s_data.get(f1_name)} (importance: {f1_val:.1%})
2. {f2_desc}: {s_data.get(f2_name)} (importance: {f2_val:.1%})
3. {f3_desc}: {s_data.get(f3_name)} (importance: {f3_val:.1%})

=== YOUR EVALUATION TASK ===
Evaluate whether opening a {user_business} in {section} is a GOOD FIT, MODERATE FIT, or POOR FIT.

=== OUTPUT FORMAT (follow exactly) ===

Sentence 1 — VERDICT:
- Start with one of:
  "Opening a **{user_business}** in {section} is a **good fit**."
  OR "...is a **moderate fit**."
  OR "...is a **poor fit**."
- Then give the single strongest reason using an actual number from the data.

Paragraph 2 — WHY (2-3 sentences):
- Explain which specific data signals support OR contradict the {user_business} idea.
- Use actual counts. Reference at least 2 numbers.
- If it is a poor or moderate fit, explain what the data actually favours instead.

Paragraph 3 — PRACTICAL ADVICE (2-3 sentences):
- If good fit: give 1-2 specific ways to position the {user_business} to maximise the section strengths, tied to actual counts.
- If moderate fit: give the one condition that would make it work, tied to a specific number.
- If poor fit: recommend what would work better in {section} based on the data, with a specific alternative business named in **bold**.

STRICT RULES:
- Bold the verdict label (**good fit**, **moderate fit**, **poor fit**)
- Bold **{user_business}** on first mention
- Bold any alternative business recommendation
- Use actual numbers in every paragraph — no vague statements
- Never mention ML, machine learning, algorithms, or confidence scores
- Write in second person, flowing prose — no bullet points, no headers
- Be honest — if the data does not support it, say so clearly and confidently"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini ERROR] {e}")

    # ── Template fallback ─────────────────────────────────────────────────────
    return (
        f"You're considering opening a **{user_business}** in **{section}**.\n\n"
        f"The data for {section} most strongly supports **{btype}** businesses. "
        f"Key signals: {f1_desc} ({s_data.get(f1_name)}), "
        f"{f2_desc} ({s_data.get(f2_name)}), "
        f"{f3_desc} ({s_data.get(f3_name)}).\n\n"
        f"Whether a {user_business} fits depends on how well it aligns with these signals."
    )


# ── Predefined responses ──────────────────────────────────────────────────────
def build_predefined_response(intent: str, text: str) -> str:
    if intent == "greeting":
        return (
            "Hello! I'm your **Shah Alam Business Advisor** 👋 — powered by geospatial "
            "data and machine learning across all 56 sections of Shah Alam.\n\n"
            "You can also include your budget, e.g. *'What business for Section 7 with RM 10,000?'* "
            "and I'll tailor the recommendation to what you can actually afford.\n\n"
            "You can also ask *'Can I open a bakery in Section 9?'* and I'll evaluate whether it's a good fit."
        )
    if intent == "help":
        return (
            "Here's what I can do:\n\n"
            "• **Business Recommendation** — e.g. *'Recommend a business for Section 14'*\n"
            "• **Budget-aware advice** — e.g. *'Section 9 with RM 20,000 budget'*\n"
            "• **Evaluate your idea** — e.g. *'Can I open a bakery in Section 7?'*\n"
            "• **Model Statistics** — ask about *accuracy* or *stats*\n"
            "• **Section List** — ask me to *list all sections*"
        )
    if intent == "stats":
        if eval_report:
            rf = eval_report.get("random_forest", eval_report)
            return (
                f"📊 **Model Performance (Random Forest + K-Fold CV)**\n\n"
                f"• CV Accuracy: **{rf.get('cv_accuracy_mean', 0):.1%}** "
                f"(±{rf.get('cv_accuracy_std', 0):.1%})\n"
                f"• Precision: **{rf.get('cv_precision_mean', 0):.1%}**\n"
                f"• Recall: **{rf.get('cv_recall_mean', 0):.1%}**\n"
                f"• F1-Score: **{rf.get('cv_f1_mean', 0):.1%}**\n\n"
                f"Final accuracy on full dataset: **{rf.get('final_accuracy', 0):.1%}**"
            )
        return "Model statistics are not available yet."
    if intent == "list_sections":
        if dataset is not None:
            sections = sorted(
                dataset["section"].dropna().astype(str).unique().tolist(),
                key=_section_sort_key,
            )
            listed = ", ".join(sections)
            return f"Shah Alam has **{len(sections)} sections**:\n\n{listed}"
        return "Section data is not loaded yet."
    return (
        "I'm here to help with business recommendations for Shah Alam! "
        "Try: *'What business should I open in Section 13?'* or "
        "*'Can I open a cafe in Section 7?'*"
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("chatbot.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data    = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    intent        = detect_intent(message)
    section       = extract_section(message)
    user_business = extract_user_business(message)

    # ── Evaluate user-proposed business ───────────────────────────────────────
    if intent == "evaluate" and section and user_business:
        prediction = predict_for_section(section)
        if prediction:
            nlg_text = generate_evaluation_response(prediction, user_business, user_message=message)
            return jsonify({
                "type":       "evaluation",
                "message":    nlg_text,
                "prediction": prediction,
            })
        return jsonify({
            "type":    "error",
            "message": f"Sorry, I don't have data for **{section}**. Please check the section name.",
        })

    # ── Standard recommendation ───────────────────────────────────────────────
    if intent == "recommend" and section:
        prediction = predict_for_section(section)
        if prediction:
            nlg_text = generate_nlg_response(prediction, user_message=message)
            return jsonify({
                "type":       "recommendation",
                "message":    nlg_text,
                "prediction": prediction,
            })
        return jsonify({
            "type":    "error",
            "message": f"Sorry, I don't have data for **{section}**. Please check the section name.",
        })

    response_text = build_predefined_response(intent, message)
    return jsonify({"type": "info", "message": response_text})


@app.route("/api/model-stats")
def model_stats():
    return jsonify(eval_report)


@app.route("/api/sections")
def sections():
    if dataset is None:
        return jsonify([])
    secs = sorted(
        dataset["section"].dropna().astype(str).unique().tolist(),
        key=_section_sort_key,
    )
    return jsonify(secs)


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)