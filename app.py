import json
import os
import pickle
import re
import requests as http_requests
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
    "population":              "number of residents",
    "food_beverage":           "F&B outlet count",
    "retail_outlet":           "retail outlet count",
    "service_business":        "service business count (salons, clinics, repair shops)",
    "entertainment":           "entertainment facility count (gyms, cinemas)",
    "educational_inst":        "educational institution count (schools, universities)",
    "corporate_office":        "corporate office count",
    "financial_inst":          "financial institution count (banks, ATMs)",
    "shopping_mall":           "retail complex & dept. store count",
    "automotive":              "automotive workshop count (car repair, gas stations)",
    "healthcare":              "healthcare facility count (hospitals, clinics, pharmacies)",
    "transportation":          "transportation node count (bus, train, transit stations)",
    "amenity_diversity_index": "amenity diversity index (Shannon index)",
}

EXPLANATION_KEYS = [
    "population", "food_beverage", "retail_outlet", "service_business",
    "entertainment", "educational_inst", "corporate_office", "financial_inst",
    "shopping_mall", "automotive", "healthcare", "transportation",
    "amenity_diversity_index",
]

# ── Business type insights ────────────────────────────────────────────────────
BUSINESS_INSIGHTS = {
    "Food & Beverage":    "High population and food culture signal strong, proven F&B demand.",
    "Retail & Commerce":  "Mall presence and retail clustering indicate a high-footfall shopping destination.",
    "Community Services": "Dense residential population with service gaps creates underserved demand.",
    "Business & Trade":   "Corporate, automotive, and financial signals indicate a trade-oriented commercial zone.",
    "Leisure & Lifestyle":"Entertainment and connectivity support lifestyle-oriented and recreational businesses.",
}

# ── Intent patterns ───────────────────────────────────────────────────────────
GREETING_PATTERNS   = [r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bsalam\b", r"\bassalamualaikum\b"]
HELP_PATTERNS       = [r"\bhelp\b", r"\bwhat can\b", r"\bwhat do you\b", r"\bwhat are you\b"]
STATS_PATTERNS      = [r"\baccuracy\b", r"\bmodel\b", r"\bperformance\b", r"\bprecision\b",
                       r"\brecall\b", r"\bf1\b", r"\bstats\b", r"\bstatistics\b"]
SECTION_LIST_PATS   = [r"\blist\s+(?:all\s+)?sections?\b", r"\bshow\s+(?:all\s+)?sections?\b",
                       r"\bwhat\s+sections?\s+(?:are\s+)?(?:available|exist)\b",
                       r"\ball\s+(?:available\s+)?(?:sections?|areas?)\b"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _section_sort_key(s: str):
    part = s.split()[-1]
    try:    return (0, int(part), "")
    except: return (1, 0, part)


def extract_all_sections(text: str) -> list[str]:
    """Extract ALL section mentions from text. Handles typos and 'Section 7 and 9' format."""
    matches = re.findall(
        r"(?:section|sektion|seksyen|seksion|sekyen)\s*(u?\d+)",
        text, re.IGNORECASE
    )
    trailing = re.findall(
        r"(?:section|sektion|seksyen|seksion|sekyen)\s*(u?\d+)\s*(?:and|or|vs|versus)\s*(u?\d+)",
        text, re.IGNORECASE
    )
    for pair in trailing:
        if pair[1] not in matches:
            matches.append(pair[1])

    seen, result = set(), []
    for m in matches:
        key = f"Section {m.upper()}"
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def extract_section(text: str) -> str | None:
    sections = extract_all_sections(text)
    return sections[0] if sections else None


def extract_budget(text: str) -> tuple[int | None, str]:
    t     = text.lower().replace(",", "")
    match = re.search(r"(?:rm|budget\s*(?:of|is)?)?\s*([0-9]+\.?[0-9]*)\s*(k\b)", t)
    if not match:
        match = re.search(r"rm\s*([0-9]+\.?[0-9]*)", t)
    if match:
        amount = float(match.group(1))
        if len(match.groups()) > 1 and match.group(2) == "k":
            amount *= 1000
        amount = int(amount)
        if amount <= 5_000:
            tier = (f"Budget: RM {amount:,} (micro). Recommend ONLY home-based, online, or mobile concepts — "
                    "no shopfront. Examples: ghost kitchen on GrabFood, online dropshipping, mobile service. "
                    "ALL ideas MUST cost under RM 5,000.")
        elif amount <= 20_000:
            tier = (f"Budget: RM {amount:,} (small). Recommend stall, kiosk, or shared-space concepts. "
                    "Examples: pasar malam stall, food court booth, home-based catering, tuition from rented room. "
                    "ALL ideas MUST be achievable under RM 20,000.")
        elif amount <= 50_000:
            tier = (f"Budget: RM {amount:,} (medium). Small shophouse or modest standalone outlet is viable. "
                    "Examples: F&B outlet, small retail shop, salon, budget franchise. "
                    "ALL ideas MUST fit within RM 50,000.")
        else:
            tier = (f"Budget: RM {amount:,} (substantial). Full shophouse, established franchise, or multi-staff. "
                    "Examples: restaurant, retail chain outlet, gym, established franchise brand.")
        return amount, tier
    return None, ""


def extract_user_business(text: str) -> str | None:
    """Extract a specific business type the user mentions or asks about."""
    cleaned = re.sub(
        r"^(?:revenue\s+potential\s+of\s+(?:a\s+|an\s+)?|"
        r"how\s+much\s+(?:can\s+i\s+earn\s+from\s+(?:a\s+|an\s+)?)?|"
        r"(?:profit|income|earning[s]?)\s+from\s+(?:a\s+|an\s+)?|"
        r"how\s+much\s+(?:does\s+|can\s+)?(?:a\s+|an\s+)?)",
        "", text.strip(), flags=re.IGNORECASE
    )

    patterns = [
        r"section\s*u?\d+\s*(?:and|or|vs|versus)\s*(?:section\s*)?u?\d+\s+for\s+(?:a\s+|an\s+)?([a-zA-Z][\w\s&]{1,40}?)\s*$",
        r"([a-zA-Z][\w\s&]{1,40}?)\s+(?:should\s+i\s+open|can\s+i\s+open|i\s+should\s+open|i\s+want\s+to\s+open)\s+in\s+section",
        r"(?:open|start|run|launch|set up|establish)\s+(?:a |an )?([a-zA-Z][\w\s&]{1,40}?)\s+(?:business|shop|store|restaurant|cafe|outlet|kiosk|stall|clinic|centre|center|studio|salon|workshop|company)\b",
        r"(?:can i|should i|i want to|i'm thinking of|thinking of opening|considering)\s+(?:open|start|run|launch)?\s*(?:a |an )?([a-zA-Z][\w\s&]{1,40}?)\s+(?:in\s+section|shop|business|store|restaurant|cafe|outlet)\b",
        r"(?:is|are)\s+(?:a |an )?([a-zA-Z][\w\s&]{1,30}?)\s+(?:good|suitable|viable|profitable|worth|bad)\s+(?:for|in)\s+section",
        r"(?:open|start)\s+(?:a |an )?([a-zA-Z][\w\s&]{1,40}?)\s+in\s+section",
        r"([a-zA-Z][\w\s&]{1,30}?)\s+(?:shop|store|restaurant|cafe|outlet|kiosk|stall|business|workshop|clinic)\s+in\s+section",
        r"(?:revenue|profit|earn|income)\s+.*?(?:open|run|start|from)\s+(?:a |an )?([a-zA-Z][\w\s&]{1,30}?)(?:\s+in\s+section|\s*$)",
        r"(?:i\s+(?:want|plan|wish|intend)\s+to\s+(?:open|start|run|sell))\s+(?:a |an )?([a-zA-Z][\w\s&]{1,40}?)(?:\s+in|\s*$)",
    ]
    stopwords = {"the", "a", "an", "my", "in", "at", "for", "with", "and", "or", "to", "is",
                 "business", "shop", "store", "section", "shah", "alam", "good", "best",
                 "open", "start", "run", "launch", "i", "can", "should", "want", "plan",
                 "from", "if", "this", "that", "which", "what", "how", "much", "any",
                 "revenue", "potential", "profit", "earn", "income", "earning", "earnings"}

    for source in [cleaned, text]:
        for pattern in patterns:
            m = re.search(pattern, source, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                words = [w for w in candidate.split() if w.lower() not in stopwords]
                if 1 <= len(words) <= 6:
                    return " ".join(words).strip()
    return None


def extract_generic_business_query(text: str) -> str | None:
    patterns = [
        r"(?:what|which)\s+section[s]?\s+(?:can\s+)?i\s+(?:can\s+)?(?:open|start|run)\s+(?:a |an )?([a-zA-Z][\w\s&]{1,40}?)\s*\??$",
        r"(?:what|which)\s+section[s]?\s+(?:can|should|is|are|would\s+be)\s+(?:best|good|suitable|ideal)?\s*(?:for|to\s+open|i\s+can\s+open)?\s+(?:a |an )?([a-zA-Z][\w\s&]{1,40}?)\s*\??$",
        r"(?:where|what\s+section)\s+(?:can|should)\s+i\s+(?:open|start|run)\s+(?:a |an )?([a-zA-Z][\w\s&]{1,40}?)\s*\??$",
        r"(?:best|good|suitable)\s+section[s]?\s+(?:for\s+)?(?:a |an )?([a-zA-Z][\w\s&]{1,40}?)\s*\??$",
        r"(?:open|start|run)\s+(?:a |an )?([a-zA-Z][\w\s&]{1,40}?)\s+(?:in\s+shah\s+alam|in\s+which\s+section)\s*\??$",
    ]
    stopwords = {"the", "a", "an", "my", "in", "at", "for", "with", "and", "or", "to", "is",
                 "business", "shop", "store", "section", "shah", "alam", "good", "best",
                 "what", "which", "where", "can", "should", "i", "open", "start", "run"}
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            words = [w for w in candidate.split() if w.lower() not in stopwords]
            if 1 <= len(words) <= 6:
                return " ".join(words).strip()
    return None


def is_comparison_query(text: str) -> bool:
    sections = extract_all_sections(text)
    if len(sections) < 2:
        return False
    comparison_words = [r"\bvs\b", r"\bversus\b", r"\bor\b", r"\bbetter\b",
                        r"\bcompare\b", r"\bwhich\s+(?:section|one)\b", r"\bbetween\b",
                        r"\band\b"]
    t = text.lower()
    return any(re.search(p, t) for p in comparison_words)


def is_revenue_or_finance_query(text: str) -> bool:
    t = text.lower()
    keywords = ["revenue", "profit", "earn", "income", "how much", "money", "return",
                "roi", "break even", "investment", "cost", "capital", "margin"]
    return any(k in t for k in keywords)


def is_market_advice_query(text: str) -> bool:
    t = text.lower()
    advice_patterns = [r"\badvice\b", r"\btips?\b", r"\bhow to\b", r"\bstrategy\b",
                       r"\bguide\b", r"\bsuggestion\b", r"\bwhat should\b"]
    return any(re.search(p, t) for p in advice_patterns)


# ── Intent detection ──────────────────────────────────────────────────────────
def detect_intent(text: str) -> str:
    t = text.lower()

    if any(re.search(p, t) for p in GREETING_PATTERNS): return "greeting"
    if any(re.search(p, t) for p in HELP_PATTERNS):     return "help"
    if any(re.search(p, t) for p in STATS_PATTERNS):    return "stats"
    if any(re.search(p, t) for p in SECTION_LIST_PATS): return "list_sections"

    sections      = extract_all_sections(text)
    user_business = extract_user_business(text)
    generic_biz   = extract_generic_business_query(text)

    if is_comparison_query(text) and len(sections) >= 2:
        return "compare"
    if is_revenue_or_finance_query(text) and sections:
        return "revenue"
    if sections and user_business:
        return "evaluate"
    if sections:
        return "recommend"
    if generic_biz:
        return "find_section"
    if user_business or is_market_advice_query(text):
        return "clarify_section"

    return "general_advice"


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
    top3_feats  = _top3_business_features(row)

    def _fmt(col, value):
        if col == "amenity_diversity_index":
            return round(float(value), 4)
        try:
            n = float(value)
            return int(n) if n.is_integer() else round(n, 4)
        except: return str(value)

    section_data = {f: _fmt(f, row[f]) for f in EXPLANATION_KEYS if f in row.index}

    return {
        "section":          section_name,
        "predicted_type":   pred_label,
        "confidence":       round(float(max(probas)) * 100, 1),
        "top3_predictions": top3_labels,
        "top3_features":    top3_feats,
        "section_data":     section_data,
    }


def _top3_business_features(row=None) -> list[tuple[str, float]]:
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

    # Global importance alone is the same for every section (it's a single
    # model-level ranking). To make the "why this section" explanation
    # section-specific, weight each feature's global importance by how much
    # of that feature THIS section actually has (normalized 0-1 against the
    # max seen across all sections in the dataset). A feature the section has
    # zero of then naturally contributes ~0 and drops out of the top 3.
    if row is not None and dataset is not None:
        contrib = {}
        for f, imp in agg_imp.items():
            try:
                val = float(row[f]) if f in row.index else 0.0
            except Exception:
                val = 0.0
            max_val = float(dataset[f].max()) if f in dataset.columns else 0.0
            norm_val = (val / max_val) if max_val > 0 else 0.0
            contrib[f] = imp * norm_val

        total = sum(contrib.values())
        if total > 0:
            agg_imp = {f: v / total for f, v in contrib.items()}
        # else: section has none of any top global feature -> fall back to
        # the global ranking computed above so we still show something sensible

    feat_imp = sorted(agg_imp.items(), key=lambda x: x[1], reverse=True)
    return [(f, round(float(i), 4)) for f, i in feat_imp[:3]]


def _build_section_profile(section_data: dict) -> str:
    return "\n".join(
        f"  - {FEATURE_DESCRIPTIONS.get(k, k)}: {v}"
        for k, v in section_data.items()
    )


def _build_multi_section_profile(sections_data: list[dict]) -> str:
    lines = []
    for pred in sections_data:
        lines.append(f"\n=== {pred['section']} DATA ===")
        lines.append(_build_section_profile(pred["section_data"]))
        lines.append(f"  → ML recommends: {pred['predicted_type']} ({pred['confidence']}% confidence)")
    return "\n".join(lines)


# ── NLG call ──────────────────────────────────────────────────────────────────
def _call_gemini(prompt: str) -> str | None:
    if not client:
        return None
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini ERROR] {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────────
# NLG PROMPTS
# ────────────────────────────────────────────────────────────────────────────────

ADVISOR_PERSONA = """You are a sharp, no-nonsense business advisor specialising in Shah Alam, Malaysia.
You give honest, specific, data-backed advice in flowing prose.
Rules that ALWAYS apply:
- Use actual numbers from the data in every paragraph , never say "strong presence" or "significant demand" without a number.
- Bold (**) every specific business name you recommend.
- Write in second person ("you", "your").
- No bullet points. No headers. No markdown lists.
- Never mention ML, machine learning, algorithms, Random Forest, or confidence scores.
- Never use marketing fluff: "ripe for", "golden opportunity", "untapped potential".
- Be direct — give a verdict, then back it with numbers.
- Keep responses to 3–4 short paragraphs maximum.
- When referencing the shopping_mall feature, always say "retail complexes and department stores", never "shopping malls".
"""


def prompt_recommend(prediction: dict, user_message: str = "") -> str:
    section      = prediction["section"]
    btype        = prediction["predicted_type"]
    top3_f       = prediction["top3_features"]
    s_data       = prediction["section_data"]
    profile      = _build_section_profile(s_data)
    budget_amt, budget_str = extract_budget(user_message)
    budget_block = f"\nBUDGET CONSTRAINT: {budget_str}\nAll ideas must fit this budget — no exceptions.\n" if budget_str else ""

    f1_desc = FEATURE_DESCRIPTIONS.get(top3_f[0][0], top3_f[0][0])
    f2_desc = FEATURE_DESCRIPTIONS.get(top3_f[1][0], top3_f[1][0])
    f3_desc = FEATURE_DESCRIPTIONS.get(top3_f[2][0], top3_f[2][0])

    return f"""{ADVISOR_PERSONA}
    {budget_block}
    === {section} GEOSPATIAL DATA ===
    {profile}

    === RECOMMENDATION TASK ===
    Business category the data points to: {btype}
    Top signals: {f1_desc} = {s_data.get(top3_f[0][0])} | {f2_desc} = {s_data.get(top3_f[1][0])} | {f3_desc} = {s_data.get(top3_f[2][0])}
    Domain insight: {BUSINESS_INSIGHTS.get(btype, "")}

    === STRICT OUTPUT FORMAT — FOLLOW EXACTLY ===

    Write one short opening paragraph (2–3 sentences) explaining why {section} suits {btype}. Use all 3 top signal numbers.

    Then output EXACTLY 3 business ideas using this format. Both tags are MANDATORY for each:

    [BUSINESS_1: Display Name]
    [SEARCH_1: simple keyword]
    4–5 sentences covering:
    1. Open with the display name in bold (**Display Name**) and what it specifically sells or offers.
    2. WHY this section suits this business — use at least 2 actual numbers from the data.
    3. Target customer using actual data (population, corporate count, student count etc).
    4. One specific operational tip — location, pricing, or differentiator.{"5. Budget: " + budget_str if budget_str else ""}

    [BUSINESS_2: Display Name]
    [SEARCH_2: simple keyword]
    4–5 sentences. DIFFERENT business type from BUSINESS_1.
    1. Open with display name in bold.
    2. Use DIFFERENT data signals from BUSINESS_1 — at least 2 numbers.
    3. Target customer with actual data.
    4. One specific operational tip.{"Budget: " + budget_str if budget_str else ""}

    [BUSINESS_3: Display Name]
    [SEARCH_3: simple keyword]
    4–5 sentences. Lower-cost or niche option vs BUSINESS_1 and BUSINESS_2.
    1. Open with display name in bold.
    2. Justify with at least 2 actual numbers.
    3. Target customer with actual data.
    4. One specific operational tip for low-cost entry.{"Budget: " + budget_str if budget_str else ""}

    End with exactly ONE sentence about the biggest risk using a specific number.

    RULES FOR TAGS — CRITICAL:
    [BUSINESS_X: Display Name] — the human-readable name shown to the user. Can be descriptive. 2–5 words.
    [SEARCH_X: simple keyword] — what gets searched on Google Maps. MUST be a simple, common, searchable noun phrase. Examples:
    - "IT support shop" NOT "Corporate IT Solutions Provider"
    - "stationery shop" NOT "Commercial Office Supplies Hub"
    - "car parts shop" NOT "Automotive Parts Wholesaler"
    - "co-working space" NOT "Section 14 WorkPod"
    - "cafe" NOT "Specialty Coffee Boutique"
    - "pharmacy" NOT "Community Health Solutions"
    Keep SEARCH tags to 2–3 common words that anyone would type into Google Maps.

    No bullet points. No sub-headers. Pure prose in each business block."""


def prompt_evaluate(prediction: dict, user_business: str, user_message: str = "") -> str:
    section  = prediction["section"]
    btype    = prediction["predicted_type"]
    top3_f   = prediction["top3_features"]
    s_data   = prediction["section_data"]
    profile  = _build_section_profile(s_data)
    budget_amt, budget_str = extract_budget(user_message)
    budget_block = f"\nBUDGET: {budget_str}\n" if budget_str else ""

    f1_desc = FEATURE_DESCRIPTIONS.get(top3_f[0][0], top3_f[0][0])
    f2_desc = FEATURE_DESCRIPTIONS.get(top3_f[1][0], top3_f[1][0])
    f3_desc = FEATURE_DESCRIPTIONS.get(top3_f[2][0], top3_f[2][0])

    return f"""{ADVISOR_PERSONA}
    {budget_block}
    === {section} GEOSPATIAL DATA ===
    {profile}

    === EVALUATION TASK ===
    The entrepreneur wants to open: **{user_business}**
    What the data most strongly supports for this section: {btype}
    Top signals: {f1_desc} = {s_data.get(top3_f[0][0])} | {f2_desc} = {s_data.get(top3_f[1][0])} | {f3_desc} = {s_data.get(top3_f[2][0])}

    === OUTPUT (3 paragraphs, no headers) ===

    Paragraph 1 — VERDICT (1–2 sentences):
    You MUST pick the verdict honestly based on this rule:
    - **good fit** → ONLY if the data category ({btype}) directly aligns with {user_business}
    - **moderate fit** → if {user_business} could work but the data points to a different category
    - **poor fit** → if the data strongly contradicts {user_business}

    For "food stalls" in a Retail & Commerce section → that is a **moderate fit** at best, not good fit.
    For "bakery" in a Food & Beverage section → that is a **good fit**.
    For "gym" in a Business & Trade section → that is a **poor fit**.

    Open with EXACTLY one of: "Opening a **{user_business}** in {section} is a **good fit**." / "...is a **moderate fit**." / "...is a **poor fit**."
    Then give the single strongest reason using an actual number.

    Paragraph 2 — DATA DEEP-DIVE (2–3 sentences):
    Which specific data signals support or contradict a {user_business} here?
    Use at least 2 actual counts. If it is a poor or moderate fit, state what the data actually supports instead.

    Paragraph 3 — ACTIONABLE ADVICE (2–3 sentences):
    Good fit → 1–2 positioning tips tied to specific counts.
    Moderate fit → the one condition that would make it work, tied to a number.
    Poor fit → name a better alternative in **bold** and explain why using the data."""


def prompt_compare(predictions: list[dict], user_business: str | None, user_message: str) -> str:
    multi_profile = _build_multi_section_profile(predictions)
    budget_amt, budget_str = extract_budget(user_message)
    budget_block = f"\nBUDGET: {budget_str}\n" if budget_str else ""

    section_names = " vs ".join(p["section"] for p in predictions)
    focus = f"specifically for a **{user_business}** business" if user_business else "for the best general business opportunity"

    return f"""{ADVISOR_PERSONA}
    {budget_block}
    === COMPARISON REQUEST ===
    The entrepreneur is comparing: {section_names}
    Focus: {focus}

    {multi_profile}

    === OUTPUT (3–4 paragraphs, no headers) ===

    Paragraph 1 — VERDICT:
    State clearly which section wins {focus} and why, using actual numbers from both sections.
    Start: "Between {section_names}, **[winning section]** is the better choice{" for a " + user_business if user_business else ""}."

    Paragraph 2 — HEAD-TO-HEAD:
    Compare the most relevant data signals side-by-side using actual counts for each section.
    {"Focus on signals that matter most for a " + user_business + " business." if user_business else ""}
    Reference at least 3 numbers from each section.

    Paragraph 3 — WINNING SECTION SPECIFICS:
    For the winning section, name 1–2 specific business concepts in **bold** that would work there.
    {"Link each concept directly to the " + user_business + " opportunity." if user_business else ""}

    Paragraph 4 — LOSING SECTION CONSOLATION (optional, only if the losing section has a clear strength):
If the losing section is genuinely better for a DIFFERENT type of business, briefly say so with a number."""


def prompt_revenue(prediction: dict, user_business: str | None, user_message: str) -> str:
    section  = prediction["section"]
    s_data   = prediction["section_data"]
    profile  = _build_section_profile(s_data)
    biz_str  = f"a **{user_business}**" if user_business else "a business"
    budget_amt, budget_str = extract_budget(user_message)
    budget_block = f"\nBUDGET: {budget_str}\n" if budget_str else ""

    return f"""{ADVISOR_PERSONA}
    {budget_block}
    === {section} GEOSPATIAL DATA ===
    {profile}

    === REVENUE / FINANCIAL ANALYSIS TASK ===
    CRITICAL: The entrepreneur is asking about {biz_str} ONLY.
    Do NOT analyse any other business type. Every paragraph must be about {biz_str} specifically.
    The entrepreneur wants to understand the revenue potential of running {biz_str} in {section}.

    === OUTPUT (3–4 paragraphs, no headers) ===

    IMPORTANT RULE: You are a business advisor, not an accountant. Do NOT give exact RM figures for revenue as a guarantee.
    Instead, reason about revenue POTENTIAL using the actual data counts as proxies for demand.

    Paragraph 1 — DEMAND POTENTIAL:
    Assess how strong the customer demand signals are for {biz_str} in {section} using actual data counts.
    Population = {s_data.get("population", "N/A")} residents, which sets the theoretical addressable market.
    State whether the demand signals are strong, moderate, or weak — and why, using specific numbers.

    Paragraph 2 — REVENUE DRIVERS:
    Which specific data signals (with actual counts) are the strongest positive drivers for revenue here?
    Explain the logical link: e.g. high corporate_office count → weekday lunch demand → higher average spend per customer.

    Paragraph 3 — REVENUE RISKS:
    Which data signals suggest revenue risk or a ceiling? Use actual counts.
    E.g. existing competition (high food_beverage count = crowded market), low footfall signals, etc.

    Paragraph 4 — REALISTIC OUTLOOK:
    Give a realistic narrative (not exact numbers) on revenue potential: is this section likely to support a
    sustainable {biz_str if user_business else "business"}? What would need to be true for it to thrive?
    {"If budget was mentioned (" + budget_str + "), factor in capital efficiency." if budget_str else ""}"""


def prompt_find_section(business_type: str, top_sections: list[dict]) -> str:
    section_summaries = []
    for p in top_sections:
        section_summaries.append(
            f"{p['section']}: population={p['section_data'].get('population')}, "
            f"ML category={p['predicted_type']}, "
            + ", ".join(f"{k}={v}" for k, v in list(p['section_data'].items())[:8])
        )
    summaries_text = "\n".join(section_summaries)

    return f"""{ADVISOR_PERSONA}

=== TASK: FIND BEST SECTIONS FOR A SPECIFIC BUSINESS ===
The entrepreneur wants to open: **{business_type}**
Below are profiles of ALL 56 sections in Shah Alam with their key geospatial data.

=== ALL SECTION DATA ===
{summaries_text}

=== OUTPUT (3 paragraphs, no headers) ===

Paragraph 1 — TOP 3 SECTIONS:
Name the 3 best sections for a **{business_type}** and give the single strongest reason for each using actual numbers.
Start: "The best sections for a **{business_type}** in Shah Alam are **[Section X]**, **[Section Y]**, and **[Section Z]**."

Paragraph 2 — WHY THESE SECTIONS WIN:
For each of the 3 sections, explain in 1–2 sentences exactly why the data supports a {business_type} there.
Use actual counts — reference at least 2 numbers per section.

Paragraph 3 — SECTIONS TO AVOID:
Name 1–2 sections that would be a poor choice for a {business_type} and explain why with actual data.
Then give a practical next step: "To get started, ask me for a full recommendation for any of these sections." """


def prompt_general_advice(user_message: str, section: str | None = None,
                          prediction: dict | None = None) -> str:
    context = ""
    if prediction:
        context = f"\n=== {section} GEOSPATIAL DATA (if relevant) ===\n{_build_section_profile(prediction['section_data'])}\n"

    return f"""{ADVISOR_PERSONA}
{context}
=== USER QUESTION ===
{user_message}

=== OUTPUT ===
Answer the user's business question directly and specifically.
If the question is about Shah Alam, use your knowledge of the area and the geospatial data if provided.
If the question is generic business advice, give a sharp, practical answer in 2–3 paragraphs.
IMPORTANT: Only use facts you are certain about. Do NOT invent specific student counts, revenue figures,
rental prices, or population numbers that are not in the geospatial data provided above.
If you genuinely cannot give a specific answer without a section number, say so clearly and ask:
"To give you a data-backed answer, which section in Shah Alam are you considering?" — but only if truly needed.
Do NOT list section names. Do NOT be vague. Do NOT give a generic MBA textbook answer.
Give real, actionable advice as a local business advisor would."""


# ── Find best sections for a business type ───────────────────────────────────
def find_best_sections_for_business(business_type: str, top_n: int = 5) -> list[dict]:
    if dataset is None:
        return []
    results = []
    for _, row in dataset.iterrows():
        section_name = row["section"]
        pred = predict_for_section(section_name)
        if pred:
            results.append(pred)
    return results


# ── Predefined responses (non-ML intents) ────────────────────────────────────
def build_predefined_response(intent: str, text: str) -> str:
    if intent == "greeting":
        return (
            "Hello! I'm your **Shah Alam Business Advisor** 👋 — backed by real geospatial data "
            "across all 56 sections of Shah Alam.\n\n"
            "Here's what I can help you with:\n\n"
            "• **Recommend a business** — *'What business should I open in Section 7?'*\n"
            "• **Evaluate your idea** — *'Can I open a bakery in Section 9?'*\n"
            "• **Compare sections** — *'Is Section 7 or Section 9 better for a car workshop?'*\n"
            "• **Find the right section** — *'Which section is best for a stationery shop?'*\n"
            "• **Revenue potential** — *'How much can I earn from a pet shop in Section 7?'*\n"
            "• **Budget-aware advice** — add *'with RM 20,000 budget'* to any question\n\n"
            "What are you thinking of opening?"
        )
    if intent == "help":
        return (
            "Here's everything I can do:\n\n"
            "• **Business Recommendation** — *'Recommend a business for Section 14'*\n"
            "• **Budget-aware advice** — *'Section 9 with RM 20,000 budget'*\n"
            "• **Evaluate your idea** — *'Can I open a bakery in Section 7?'*\n"
            "• **Compare two sections** — *'Section 7 or Section 9 for a car workshop?'*\n"
            "• **Find best section** — *'Which section is best for a stationery shop?'*\n"
            "• **Revenue analysis** — *'Revenue potential of a pet shop in Section 7?'*\n"
            "• **Model stats** — ask about *accuracy* or *model performance*\n"
            "• **Section list** — *'List all sections'*"
        )
    if intent == "stats":
        if eval_report:
            rf = eval_report.get("random_forest", eval_report)
            return (
                f"📊 **Model Performance (Random Forest + 5-Fold Stratified CV)**\n\n"
                f"• CV Accuracy: **{rf.get('cv_accuracy_mean', 0):.1%}** "
                f"(±{rf.get('cv_accuracy_std', 0):.1%})\n"
                f"• CV Precision: **{rf.get('cv_precision_mean', 0):.1%}**\n"
                f"• CV Recall: **{rf.get('cv_recall_mean', 0):.1%}**\n"
                f"• CV F1-Score: **{rf.get('cv_f1_mean', 0):.1%}**\n\n"
                f"Final accuracy on full dataset: **{rf.get('final_accuracy', 0):.1%}**\n\n"
                f"Top feature by importance: **corporate_office** ({rf.get('feature_importance', {}).get('corporate_office', 0):.1%})"
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
    return None


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
    sections      = extract_all_sections(message)
    section       = sections[0] if sections else None
    user_business = extract_user_business(message)
    generic_biz   = extract_generic_business_query(message)

    predefined = build_predefined_response(intent, message)
    if predefined is not None:
        return jsonify({"type": "info", "message": predefined})

    if intent == "compare" and len(sections) >= 2:
        predictions = [predict_for_section(s) for s in sections[:3]]
        predictions = [p for p in predictions if p]
        if predictions:
            prompt   = prompt_compare(predictions, user_business, message)
            nlg_text = _call_gemini(prompt)
            if not nlg_text:
                names = " and ".join(p["section"] for p in predictions)
                nlg_text = f"I have data for {names}. Ask me to recommend individually for each section."
            return jsonify({
                "type":       "comparison",
                "message":    nlg_text,
                "prediction": predictions[0],
            })
        return jsonify({"type": "error",
                        "message": "I couldn't find data for those sections. Please check the section names."})

    if intent == "revenue" and section:
        prediction = predict_for_section(section)
        if prediction:
            prompt   = prompt_revenue(prediction, user_business, message)
            nlg_text = _call_gemini(prompt)
            if not nlg_text:
                nlg_text = (f"**{section}** has a population of "
                            f"{prediction['section_data'].get('population', 'N/A')} residents. "
                            f"The data most strongly supports **{prediction['predicted_type']}** "
                            f"businesses here — a good starting signal for revenue potential.")
            return jsonify({
                "type":       "revenue",
                "message":    nlg_text,
                "prediction": prediction,
            })

    if intent == "evaluate" and section and user_business:
        prediction = predict_for_section(section)
        if prediction:
            prompt   = prompt_evaluate(prediction, user_business, message)
            nlg_text = _call_gemini(prompt)
            if not nlg_text:
                nlg_text = (f"You're considering a **{user_business}** in **{section}**. "
                            f"The data most strongly supports **{prediction['predicted_type']}** here. "
                            f"Ask me to compare or get a full recommendation.")
            return jsonify({
                "type":         "evaluation",
                "message":      nlg_text,
                "prediction":   prediction,
                "user_business": user_business,   
            })
        return jsonify({"type": "error",
                        "message": f"I don't have data for **{section}**. Please check the section name."})

    if intent == "recommend" and section:
        prediction = predict_for_section(section)
        if prediction:
            prompt   = prompt_recommend(prediction, message)
            nlg_text = _call_gemini(prompt)
            if not nlg_text:
                nlg_text = (f"For **{section}**, the data points to **{prediction['predicted_type']}** "
                            f"as the best business category. "
                            f"{BUSINESS_INSIGHTS.get(prediction['predicted_type'], '')}")
            return jsonify({
                "type":       "recommendation",
                "message":    nlg_text,
                "prediction": prediction,
            })
        return jsonify({"type": "error",
                        "message": f"I don't have data for **{section}**. Please check the section name."})

    if intent == "find_section" and generic_biz:
        all_predictions = find_best_sections_for_business(generic_biz)
        if all_predictions:
            prompt   = prompt_find_section(generic_biz, all_predictions)
            nlg_text = _call_gemini(prompt)
            if not nlg_text:
                nlg_text = (f"To find the best section for a **{generic_biz}**, try asking: "
                            f"*'Recommend a business for Section 7'* and compare a few sections manually.")
            return jsonify({"type": "find_section", "message": nlg_text})
        return jsonify({"type": "error", "message": "Could not load section data."})

    if intent == "clarify_section":
        biz_hint   = user_business or "that type of business"
        prediction = predict_for_section(section) if section else None
        prompt     = prompt_general_advice(message, section, prediction)
        nlg_text   = _call_gemini(prompt)
        if not nlg_text:
            nlg_text = (
                f"Great question about **{biz_hint}**! To give you a data-backed answer, "
                f"which section in Shah Alam are you considering?\n\n"
                f"For example: *'Is a {biz_hint} good for Section 7?'* or "
                f"*'Which section is best for a {biz_hint}?'*"
            )
        return jsonify({"type": "clarify", "message": nlg_text, "prediction": prediction})

    prediction = predict_for_section(section) if section else None
    prompt     = prompt_general_advice(message, section, prediction)
    nlg_text   = _call_gemini(prompt)
    if not nlg_text:
        nlg_text = (
            "I'm your Shah Alam Business Advisor — ask me anything about opening a business here!\n\n"
            "Try: *'What business for Section 7?'*, *'Which section is best for a café?'*, "
            "or *'Compare Section 9 vs Section 14 for a pharmacy.'*"
        )
    return jsonify({"type": "advice", "message": nlg_text, "prediction": prediction})


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


@app.route("/api/competitors")
def competitors():
    MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not MAPS_KEY:
        return jsonify({"error": "Google Maps API key not set", "results": []})

    lat           = request.args.get("lat", type=float)
    lng           = request.args.get("lng", type=float)
    business_type = request.args.get("business_type", "Food & Beverage")
    keyword       = request.args.get("keyword", "")
    limit         = request.args.get("limit", 8, type=int)

    if not lat or not lng:
        return jsonify({"error": "lat and lng required", "results": []})

    TYPE_MAP = {
        "Food & Beverage":    "restaurant",
        "Retail & Commerce":  "shopping_mall",
        "Business & Trade":   "finance",
        "Community Services": "school",
        "Leisure & Lifestyle":"gym",
    }
    place_type = TYPE_MAP.get(business_type, "establishment")

    def simplify_keyword(kw):
        words = kw.strip().split()
        if len(words) >= 3:
            return " ".join(words[-2:])
        return kw

    search_keyword = simplify_keyword(keyword) if keyword else ""

    try:
        search_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

        def do_search(radius):
            params = {"location": f"{lat},{lng}", "radius": radius, "key": MAPS_KEY}
            if search_keyword:
                params["keyword"] = search_keyword
            else:
                params["type"] = place_type
            return http_requests.get(search_url, params=params, timeout=8).json()

        data = do_search(1000)
        print(f"[Competitors] status={data.get('status')} results={len(data.get('results',[]))} keyword='{search_keyword}' radius=1000")

        if len(data.get("results", [])) < 3:
            data2 = do_search(2000)
            if len(data2.get("results", [])) > len(data.get("results", [])):
                data = data2

        if len(data.get("results", [])) == 0 and search_keyword:
            data3_params = {"location": f"{lat},{lng}", "radius": 2000, "type": place_type, "key": MAPS_KEY}
            data3 = http_requests.get(search_url, params=data3_params, timeout=8).json()
            if len(data3.get("results", [])) > 0:
                data = data3

        results = []
        for place in data.get("results", [])[:limit]:
            photo_url = None
            photos = place.get("photos", [])
            if photos:
                ref = photos[0].get("photo_reference", "")
                if ref:
                    photo_url = (f"https://maps.googleapis.com/maps/api/place/photo"
                                 f"?maxwidth=400&photo_reference={ref}&key={MAPS_KEY}")
            results.append({
                "name":          place.get("name", ""),
                "address":       place.get("vicinity", ""),
                "rating":        place.get("rating", None),
                "total_ratings": place.get("user_ratings_total", 0),
                "open_now":      place.get("opening_hours", {}).get("open_now", None),
                "photo_url":     photo_url,
                "place_id":      place.get("place_id", ""),
            })
        return jsonify({"results": results, "business_type": business_type})

    except Exception as e:
        print(f"[Competitors API Error] {e}")
        return jsonify({"error": str(e), "results": []})


# ── NEW: Demand sources for evaluation responses ──────────────────────────────
@app.route("/api/demand-sources")
def demand_sources():
    MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not MAPS_KEY:
        return jsonify({"error": "Google Maps API key not set", "results": []})

    lat     = request.args.get("lat", type=float)
    lng     = request.args.get("lng", type=float)
    section = request.args.get("section", "")

    if not lat or not lng:
        return jsonify({"error": "lat and lng required", "results": []})

    # Each tuple: (search keyword, display category, emoji)
    DEMAND_QUERIES = [
        ("university",      "University",       "🎓"),
        ("college",         "College",          "🎓"),
        ("school",          "School",           "🏫"),
        ("office building", "Corporate Office", "🏢"),
        ("LRT station",     "Transit Station",  "🚉"),
        ("hospital",        "Hospital",         "🏥"),
        ("shopping mall",   "Shopping Mall",    "🛍️"),
    ]

    search_url  = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    all_results = []
    seen_ids    = set()

    try:
        for keyword, category, emoji in DEMAND_QUERIES:
            params = {
                "location": f"{lat},{lng}",
                "radius":   2500,
                "keyword":  keyword,
                "key":      MAPS_KEY,
            }
            data = http_requests.get(search_url, params=params, timeout=8).json()
            for place in data.get("results", [])[:2]:   # max 2 per category
                pid = place.get("place_id", "")
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                photo_url = None
                photos = place.get("photos", [])
                if photos:
                    ref = photos[0].get("photo_reference", "")
                    if ref:
                        photo_url = (f"https://maps.googleapis.com/maps/api/place/photo"
                                     f"?maxwidth=400&photo_reference={ref}&key={MAPS_KEY}")
                all_results.append({
                    "name":      place.get("name", ""),
                    "address":   place.get("vicinity", ""),
                    "rating":    place.get("rating", None),
                    "category":  category,
                    "emoji":     emoji,
                    "photo_url": photo_url,
                    "place_id":  pid,
                })

        return jsonify({"results": all_results[:12], "section": section})

    except Exception as e:
        print(f"[Demand Sources API Error] {e}")
        return jsonify({"error": str(e), "results": []})


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)