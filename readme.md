# 🏙️ AN INTELLIGENT BUSINESS RECOMMENDATION INTEGRATING GEOSPATIAL DATA AND CHATBOT ASSISTANCE 

> FYP Project - MUHAMMAD EZAM HAZAMI BIN SUHAINI (2023298984)
> Universiti Teknologi Mara | Bachelor of Computer Science (Hons.) | October 2025 - Present

This System supports :
- **Geospatial Data**, Google Maps API with 56 sections in Shah Alam
- **Machine Learning Model**, multi-class classification with KFold cross validation
- **Hybrid Chabot**, mix of intention detection + predefined answers + Google Gemini API NLG

--

## 📁 Project Structure 

#Steps
1) **.env** - Assign API keys for google maps api, google gemini api 
2) **requirement.txt** - download dependencies
3) **collect_data.py** -  Extract geospatial data from Google Maps, initialize latitude and longitude for each sections, mappings feature types, saved data as shahAlamBusiness.csv
4) **label_data.py** - Assign business_type label for machine learning classification x,y.
5) **train_model.py** - Train machine learning model using Geospatial dataset, saved the model evaluation_report, label_encoder, feature_names as .pkl
6) **app.py** - load model artefacts, feature description explainability, business_insight use to support arguments, intent detection, chatbot integrations

--

## 🗂️ Full Pipeline (With real Google Maps data)

### Step 1 — Setup environment
```bash
pip install -r requirements.txt
cp .env
# Edit .env and set GOOGLE_MAPS_API_KEY and optionally GEMINI_API_KEY
```

### Step 2 — Collect geospatial data
```bash
python collect_data.py
# Queries Google Maps Places API for all 56 sections
# Output: data/shah_alam_geospatial.csv  (~30-60 min due to API rate limits)
```

### Step 3 — Label the dataset
```bash
python label_data.py
# Applies rule-based heuristics to assign business_type labels
# Output: data/shah_alam_labeled.csv
```

### Step 4 — Train the model
```bash
python train_model.py
# Runs 5-Fold CV + RandomizedSearchCV hyperparameter tuning
# Output: models/rf_model.pkl + evaluation_report.json
```

### Step 5 — Launch the app
```bash
python app.py
```

## 🤖 Machine Learning Details

| Component | Detail |
|-----------|--------|
| Algorithm | Random Forest Classifier |
| Problem type | Multi-class classification (9 classes) |
| Validation | 5-Fold Cross Validation |
| Tuning | RandomizedSearchCV (30 iterations) |
| Metrics | Accuracy, Precision, Recall, F1-Score (macro) |
| CV Accuracy | ~87% on synthetic dataset |

### Business Type Classes
1. Food & Beverage
2. Retail
3. Professional Services
4. Entertainment
5. Health & Wellness
6. Education Support
7. Automotive
8. Financial Services
9. Hospitality

### Features (F1–F12)
| ID | Feature | Description |
|----|---------|-------------|
| F1 | food_beverage | Number of F&B outlets |
| F2 | retail_outlet | Retail store density |
| F3 | service_business | Salons, clinics, repair shops |
| F4 | entertainment | Gyms, cinemas, theme parks |
| F5 | educational_inst | Schools and universities |
| F6 | residential_area | Residential density proxy |
| F7 | corporate_office | Office buildings |
| F8 | financial_inst | Banks and ATMs |
| F9 | hotel | Hotels and accommodation |
| F10 | shopping_mall | Shopping malls |
| F11 | workshop | Automotive workshops |
| F12 | amenity_diversity_index | Shannon diversity of amenities |

---

## 💬 Chatbot API

### POST /api/recommend
```json
{ "message": "What business should I open in Section 7?" }
```
Response:
```json
{
  "type": "recommendation",
  "message": "Based on my geospatial analysis of Section 7, I recommend...",
  "prediction": {
    "section": "Section 7",
    "predicted_type": "Food & Beverage",
    "confidence": 87.3,
    "top3_predictions": [["Food & Beverage", 87.3], ["Retail", 8.1], ...],
    "top3_features": [["food_beverage", 0.1304], ...],
    "section_data": { "food_beverage": 45, ... }
  }
}
```

### GET /api/model-stats
Returns CV accuracy, precision, recall, F1, best hyperparameters, and feature importances.

### GET /api/sections
Returns list of all 56 section names.

---

## 🌐 NLG Integration

The chatbot uses a **hybrid approach**:
1. **Intent Detection** — regex-based classifier (greeting / help / stats / recommend)
2. **Predefined Answers** — verified facts from feature importance data
3. **NLG (Google Gemini)** — expands predefined context into natural language

Set `GEMINI_API_KEY` in `.env` to enable Gemini NLG. If not set, the system falls back to high-quality template-based responses.

---

## 📊 Sample Chatbot Interactions

| User Input | Intent | Response |
|-----------|--------|----------|
| "Hi!" | greeting | Welcome message with instructions |
| "Best business for Section 7?" | recommend | ML prediction + Gemini explanation |
| "Show accuracy" | stats | CV metrics display |
| "List all sections" | list_sections | All 56 sections |
| "What can you do?" | help | Feature list |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11 + Flask 3.0
- **ML**: scikit-learn (RandomForestClassifier, KFold, RandomizedSearchCV)
- **Data**: Google Maps Places API, Pandas, NumPy
- **NLG**: Google Gemini 1.5 Flash API
- **Frontend**: Vanilla HTML/CSS/JS (no framework)

---



