"""
train_model.py
--------------
Trains Random Forest and Logistic Regression classifiers on the Shah Alam
geospatial dataset using Stratified K-Fold cross-validation and hyperparameter
tuning (Random Forest only).

Outputs:
  models/rf_model.pkl              — best tuned RandomForest pipeline
  models/lr_model.pkl              — LogisticRegression pipeline
  models/label_encoder.pkl         — LabelEncoder for business_type
  models/feature_names.pkl         — ordered feature list (all columns used)
  models/evaluation_report.json    — accuracy, precision, recall, f1 for both

Usage:
    python train_model.py
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import randint

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

LABELED_PATH = "data/shah_alam_labeled2.csv"
TARGET = "business_type"
MODEL_FEATURES = [
    "population",
    "food_beverage",
    "retail_outlet",
    "service_business",
    "entertainment",
    "educational_inst",
    "corporate_office",
    "financial_inst",
    "shopping_mall",
    "automotive",
    "healthcare",
    "transportation",
    "amenity_diversity_index",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if os.path.exists(LABELED_PATH):
        print(f"📂  Loading dataset: {LABELED_PATH}")
        return pd.read_csv(LABELED_PATH)
    raise FileNotFoundError(
        f"Dataset not found at '{LABELED_PATH}'. "
        "Please make sure the CSV is in the data/ folder."
    )


def build_pipeline(classifier) -> Pipeline:
    """
    Returns a full sklearn Pipeline that:
      1. One-hot encodes all categorical/object columns
      2. Passes through numeric columns as-is
      3. Applies the given classifier
    """
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, _categorical_cols)
        ],
        remainder="passthrough"
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])


# ── Main training routine ─────────────────────────────────────────────────────

def train():
    df = load_data()
    print(f"   {len(df)} rows, {df[TARGET].nunique()} classes")
    print(f"   Class distribution:\n{df[TARGET].value_counts().to_string()}\n")

    # ── Feature / target split (mirrors notebook: drop target, keep everything else)
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[MODEL_FEATURES].copy()
    y_raw = df[TARGET]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Identify categorical columns for the preprocessor (used inside build_pipeline)
    global _categorical_cols
    _categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    feature_names = X.columns.tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. RANDOM FOREST — baseline CV
    # ══════════════════════════════════════════════════════════════════════════
    print("🌲  Random Forest — 5-Fold Stratified CV (baseline)…")
    rf_pipeline = build_pipeline(RandomForestClassifier(random_state=42))

    rf_cv = cross_validate(
        rf_pipeline, X, y, cv=skf,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        return_train_score=True,
    )

    rf_cv_acc   = rf_cv["test_accuracy"].mean()
    rf_cv_prec  = rf_cv["test_precision_macro"].mean()
    rf_cv_rec   = rf_cv["test_recall_macro"].mean()
    rf_cv_f1    = rf_cv["test_f1_macro"].mean()

    print(f"   All Folds Accuracy : {rf_cv['test_accuracy']}")
    print(f"   Mean Accuracy      : {rf_cv_acc * 100:.2f}%")
    print(f"   CV Precision       : {rf_cv_prec:.4f}")
    print(f"   CV Recall          : {rf_cv_rec:.4f}")
    print(f"   CV F1              : {rf_cv_f1:.4f}\n")

    # ── Hyperparameter tuning ─────────────────────────────────────────────────
    print("⚙️   Random Forest — RandomizedSearchCV hyperparameter tuning…")
    param_dist = {
        "model__n_estimators":      [50, 100, 200, 300],
        "model__max_depth":         [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 4, 6, 8],
        "model__min_samples_leaf":  [1, 2, 3],
        "model__max_features":      ["sqrt", "log2", None],
        "model__bootstrap":         [True, False],
        'model__class_weight':      ['balanced', 'balanced_subsample', None],
    }

    rf_search = RandomizedSearchCV(
        build_pipeline(RandomForestClassifier(random_state=42)),
        param_distributions=param_dist,
        n_iter=30,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    rf_search.fit(X, y)

    print(f"   Best Accuracy : {rf_search.best_score_ * 100:.2f}%")
    print(f"   Best Params   : {rf_search.best_params_}\n")

    best_rf_pipeline = rf_search.best_estimator_

    best_cv = cross_validate(
        best_rf_pipeline, X, y, cv=skf,
        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        return_train_score=True
    )
    y_pred_best = cross_val_predict(best_rf_pipeline, X, y, cv=skf)

    rf_cv_acc_best   = best_cv["test_accuracy"].mean()
    rf_cv_prec_best  = best_cv["test_precision_macro"].mean()
    rf_cv_rec_best   = best_cv["test_recall_macro"].mean()
    rf_cv_f1_best    = best_cv["test_f1_macro"].mean()

    # Final fit & evaluation on full data
    best_rf_pipeline.fit(X, y)                         # train on ALL 56 rows
    y_pred_full = best_rf_pipeline.predict(X)          # predict ALL 56 rows
    rf_acc    = accuracy_score(y, y_pred_full)         # this is the real final accuracy
    rf_report = classification_report(y, y_pred_full, target_names=le.classes_,
                                    output_dict=True, zero_division=0)

    print("📊  RF Final Evaluation (full dataset fit):")
    print(f"   Accuracy : {rf_acc:.4f}")
    print(classification_report(y, y_pred_full, target_names=le.classes_))

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Feature importances (RF only)
    # ══════════════════════════════════════════════════════════════════════════
    rf_model        = best_rf_pipeline.named_steps["model"]
    rf_preprocessor = best_rf_pipeline.named_steps["preprocessor"]

    # Recover feature names using the fitted preprocessor directly.
    # get_feature_names_out() returns OHE-expanded names for categorical cols
    # and the original column names for numeric passthrough cols.
    try:
        all_feat = rf_preprocessor.get_feature_names_out().tolist()
        # Strip sklearn's "cat__" / "remainder__" prefixes for readability
        all_feat = [
            n.replace("cat__onehot__", "")
             .replace("remainder__", "")
             .replace("cat__", "")
            for n in all_feat
        ]
    except Exception as e:
        print(f"   ⚠️  Could not recover feature names: {e}")
        all_feat = [f"feature_{i}" for i in range(len(rf_model.feature_importances_))]
    importances = rf_model.feature_importances_
    feat_imp    = sorted(zip(all_feat, importances), key=lambda x: x[1], reverse=True)

    print("🌲  Top 15 Feature Importances (RF):")
    for feat, imp in feat_imp[:15]:
        bar = "█" * int(imp * 50)
        print(f"   {feat:<40} {imp:.4f}  {bar}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Save artefacts
    # ══════════════════════════════════════════════════════════════════════════
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(best_rf_pipeline, f)

    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    eval_report = {
        "random_forest": {
            "cv_accuracy_mean":  float(rf_cv_acc_best),
            "cv_accuracy_std":   float(best_cv["test_accuracy"].std()),
            "cv_precision_mean": float(rf_cv_prec_best),
            "cv_recall_mean":    float(rf_cv_rec_best),
            "cv_f1_mean":        float(rf_cv_f1_best),
            "best_params":       rf_search.best_params_,
            "final_accuracy":    float(rf_acc),
            "feature_importance": {f: float(i) for f, i in feat_imp},
            "classification_report": rf_report,
        }
    }

    with open("models/evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    print("\n✅  All artefacts saved to models/")
    print("   rf_model2.pkl")
    print("   label_encoder2.pkl")
    print("   feature_names2.pkl")
    print("   evaluation_report2.json")


if __name__ == "__main__":
    train()

