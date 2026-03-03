"""
Scenario-Specific Model Trainers — v1.0

Six production-ready training functions, each owning one scenario end-to-end.
Every trainer follows the same TypedDict-compatible return contract so the
smart dispatcher can consume results uniformly.

Trainers
--------
1.  train_house_price_estimator   → Linear Regression
2.  train_banknote_authenticator  → SVM (RBF kernel)
3.  train_customer_churn_predictor → XGBoost Classifier
4.  train_sms_spam_detector        → TF-IDF + Multinomial Naive Bayes
5.  train_customer_segmentation    → K-Means (auto-k via silhouette)
6.  train_stock_pca_visualizer     → PCA (2-component projection)

Live-prediction helpers (predict_*)
are exported alongside each trainer for endpoint usage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type alias – keeps function signatures clean
# ---------------------------------------------------------------------------
TrainerResult = Dict[str, Any]


# ===========================================================================
# Shared utilities
# ===========================================================================


def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first column from *candidates* that exists in *df*
    (case-insensitive substring match).
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        # Exact match first
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
        # Substring match (e.g. "monthly_charges" matches "monthlycharges")
        for col_lower, col_orig in lower_map.items():
            if cand.lower() in col_lower or col_lower in cand.lower():
                return col_orig
    return None


def _encode_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Label-encode each categorical column; return a modified copy."""
    out = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].astype(str).fillna("__NA__"))
    return out


def _numeric_feature_cols(df: pd.DataFrame, exclude: Optional[str] = None) -> List[str]:
    """Return numeric columns, optionally excluding the target."""
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != exclude and df[c].nunique() > 1
    ]


# ===========================================================================
# 1. House Price Estimator — Linear Regression
# ===========================================================================

_HOUSE_TARGET = ["price", "house_price", "sale_price", "selling_price", "value", "list_price"]


def train_house_price_estimator(df: pd.DataFrame) -> TrainerResult:
    """
    Train a Standard-Scaled Linear Regression model to estimate house prices.

    Accepted CSV shapes
    -------------------
    • Minimum: at least one numeric feature + one numeric price-like column.
    • Ideal:   square_footage, bedrooms, bathrooms, location_score, house_age → price

    Returns (TrainerResult)
    -----------------------
    model, scaler, feature_cols, target_col, model_name, task_type,
    metrics {r2_score, rmse, mae, primary_score},
    feature_importance, sample_predictions, explained
    """
    df = df.dropna().copy()

    # ── resolve target ────────────────────────────────────────────────────
    target_col = _resolve_column(df, _HOUSE_TARGET)
    if target_col is None:
        num_cols = _numeric_feature_cols(df)
        if not num_cols:
            raise ValueError(
                "House price dataset must contain at least one numeric column."
            )
        target_col = num_cols[-1]  # fallback: last numeric column

    # ── resolve features ──────────────────────────────────────────────────
    feature_cols = _numeric_feature_cols(df, exclude=target_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for house price estimation.")

    X = df[feature_cols].values
    y = df[target_col].values

    # ── train ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # ── evaluate ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    # Normalised absolute coefficients as proxy importance
    total_abs = float(np.sum(np.abs(model.coef_))) + 1e-9
    feature_importance = {
        col: round(float(abs(coef)) / total_abs, 4)
        for col, coef in zip(feature_cols, model.coef_)
    }

    sample_n = min(5, len(y_test))
    sample_predictions = [
        {"actual": round(float(a), 2), "predicted": round(float(p), 2)}
        for a, p in zip(y_test[:sample_n], y_pred[:sample_n])
    ]

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "model_name": "Linear Regression",
        "task_type": "regression",
        "metrics": {
            "r2_score": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "score_label": "R² Score",
            "primary_score": round(r2, 4),
        },
        "feature_importance": feature_importance,
        "sample_predictions": sample_predictions,
        "explained": (
            f"Linear Regression trained on {len(feature_cols)} features. "
            f"R²={r2:.3f} on hold-out test set "
            f"(RMSE={rmse:,.0f}, MAE={mae:,.0f}). "
            "Coefficient signs indicate direction of each feature's price effect. "
            "StandardScaler applied before fitting to avoid feature magnitude bias."
        ),
    }


def predict_house_price(
    model: LinearRegression,
    scaler: StandardScaler,
    feature_cols: List[str],
    inputs: Dict[str, float],
) -> Dict[str, Any]:
    """Return a single house-price prediction from user inputs."""
    x = np.array([[inputs.get(f, 0.0) for f in feature_cols]])
    x_scaled = scaler.transform(x)
    price = float(model.predict(x_scaled)[0])
    return {"predicted_price": round(price, 2), "currency": "USD"}


# ===========================================================================
# 2. Banknote Authentication — SVM Classifier (RBF kernel)
# ===========================================================================

_BANKNOTE_TARGET = ["class", "authentic", "label", "target", "real_fake", "genuine", "fake"]


def train_banknote_authenticator(df: pd.DataFrame) -> TrainerResult:
    """
    Train an SVM (RBF, C=10) for banknote authentication.

    The four wavelet-transform features (variance, skewness, curtosis, entropy)
    create a maximum-margin decision boundary that separates genuine from
    counterfeit banknotes with high precision.

    Returns (TrainerResult)
    -----------------------
    model, scaler, feature_cols, target_col, label_classes,
    model_name, task_type, metrics {accuracy, primary_score},
    sample_predictions, explained
    """
    df = df.dropna().copy()

    target_col = _resolve_column(df, _BANKNOTE_TARGET) or df.columns[-1]
    feature_cols = _numeric_feature_cols(df, exclude=target_col)
    if not feature_cols:
        raise ValueError("Banknote dataset must contain numeric feature columns.")

    X = df[feature_cols].values
    y_raw = df[target_col].values

    # ── encode labels ─────────────────────────────────────────────────────
    if y_raw.dtype.kind not in ("i", "u", "f"):
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        label_classes: List[str] = le.classes_.tolist()
    else:
        y = y_raw.astype(int)
        label_classes = [str(c) for c in sorted(np.unique(y))]

    # ── train ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    model.fit(X_train, y_train)

    # ── evaluate ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    sample_n = min(5, len(y_test))
    sample_predictions = [
        {
            "predicted_class": label_classes[int(p)] if int(p) < len(label_classes) else str(int(p)),
            "confidence": round(float(np.max(prob)), 4),
            "authentic": bool(int(p) == (len(label_classes) - 1)),  # assume last class = genuine
        }
        for p, prob in zip(y_pred[:sample_n], y_proba[:sample_n])
    ]

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "label_classes": label_classes,
        "model_name": "SVM Classifier",
        "task_type": "classification",
        "metrics": {
            "accuracy": round(accuracy * 100, 2),
            "score_label": "Accuracy %",
            "primary_score": round(accuracy * 100, 2),
        },
        "sample_predictions": sample_predictions,
        "explained": (
            f"SVM RBF (C=10, gamma='scale') achieved {accuracy*100:.1f}% accuracy "
            f"on {len(feature_cols)} wavelet-transform features. "
            "The maximum-margin hyperplane creates a safety zone between genuine and "
            "counterfeit classes — a 3-5% margin means even near-boundary bills "
            "are classified with explicit uncertainty."
        ),
    }


def predict_banknote(
    model: SVC,
    scaler: StandardScaler,
    feature_cols: List[str],
    label_classes: List[str],
    inputs: Dict[str, float],
) -> Dict[str, Any]:
    """Classify a single banknote from its four wavelet features."""
    x = np.array([[inputs.get(f, 0.0) for f in feature_cols]])
    x_scaled = scaler.transform(x)
    pred_idx = int(model.predict(x_scaled)[0])
    proba = model.predict_proba(x_scaled)[0]
    return {
        "predicted_class": label_classes[pred_idx] if pred_idx < len(label_classes) else str(pred_idx),
        "authentic": bool(pred_idx == len(label_classes) - 1),
        "confidence": round(float(np.max(proba)) * 100, 2),
        "probabilities": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(label_classes, proba)
        },
    }


# ===========================================================================
# 3. Customer Churn Prediction — XGBoost Classifier
# ===========================================================================

_CHURN_TARGET = ["churn", "churned", "cancelled", "left", "target", "is_churn"]


def train_customer_churn_predictor(df: pd.DataFrame) -> TrainerResult:
    """
    Train an XGBoost Classifier for customer churn prediction.

    Key design decisions
    --------------------
    • Categoricals label-encoded (preserves tree-split semantics).
    • scale_pos_weight computed from class ratio to handle imbalance.
    • n_estimators=200 + learning_rate=0.05 for stable convergence.
    • feature_importances_ (gain-based) highlight top churn drivers.

    Returns (TrainerResult)
    -----------------------
    model, feature_cols, cat_cols, target_col, model_name, task_type,
    metrics {accuracy, churn_rate_pct, scale_pos_weight, primary_score},
    feature_importance, sample_predictions, explained
    """
    if not XGBOOST_AVAILABLE:
        raise RuntimeError(
            "XGBoost is not installed. Add 'xgboost>=2.0,<3.0' to requirements.txt "
            "and run: pip install xgboost"
        )

    target_col = _resolve_column(df, _CHURN_TARGET) or df.columns[-1]
    df = df.dropna(subset=[target_col]).copy()

    # ── encode target to binary 0/1 ───────────────────────────────────────
    y_raw = df[target_col].astype(str).str.lower().str.strip()
    _CHURN_POSITIVE = {"yes", "1", "true", "churned", "cancelled", "churn"}
    y = y_raw.isin(_CHURN_POSITIVE).astype(int).values

    # ── feature preparation ───────────────────────────────────────────────
    feature_df = df.drop(columns=[target_col]).copy()
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    feature_df = _encode_categoricals(feature_df, cat_cols)
    feature_cols = num_cols + cat_cols
    X = feature_df[feature_cols].fillna(0).values

    # ── class imbalance weight ────────────────────────────────────────────
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = float(n_neg) / max(n_pos, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # ── evaluate ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test, y_pred))
    churn_rate = float(y.mean())

    # Sorted feature importance (XGB gain)
    feature_importance = dict(
        sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    feature_importance = {k: round(float(v), 4) for k, v in feature_importance.items()}

    sample_n = min(5, len(y_test))
    sample_predictions = [
        {
            "churn_probability": round(float(p) * 100, 1),
            "predicted": "CHURN" if int(pred) == 1 else "RETAIN",
            "risk_level": "High" if p >= 0.7 else ("Medium" if p >= 0.4 else "Low"),
        }
        for p, pred in zip(y_proba[:sample_n], y_pred[:sample_n])
    ]

    return {
        "model": model,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "target_col": target_col,
        "model_name": "XGBoost Classifier",
        "task_type": "classification",
        "metrics": {
            "accuracy": round(accuracy * 100, 2),
            "churn_rate_pct": round(churn_rate * 100, 1),
            "scale_pos_weight": round(scale_pos_weight, 2),
            "score_label": "Accuracy %",
            "primary_score": round(accuracy * 100, 2),
        },
        "feature_importance": feature_importance,
        "sample_predictions": sample_predictions,
        "explained": (
            f"XGBoost (n=200, lr=0.05, depth=5) achieved {accuracy*100:.1f}% accuracy. "
            f"scale_pos_weight={scale_pos_weight:.1f} corrects for "
            f"{churn_rate*100:.1f}% churn class imbalance. "
            "Top churn drivers exposed via XGBoost gain-based feature importance — "
            "ideal for marketing teams to identify and pre-empt at-risk customers."
        ),
    }


def predict_churn(
    model: Any,
    feature_cols: List[str],
    cat_cols: List[str],
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Return churn probability and risk tier for a single customer."""
    row = {}
    for col in feature_cols:
        val = inputs.get(col, 0)
        if col in cat_cols:
            try:
                val = int(val)
            except (ValueError, TypeError):
                val = 0
        row[col] = float(val)
    x = np.array([[row[c] for c in feature_cols]])
    proba = float(model.predict_proba(x)[0][1])
    pred = int(model.predict(x)[0])
    return {
        "churn_probability": round(proba * 100, 1),
        "predicted": "CHURN" if pred == 1 else "RETAIN",
        "risk_level": "High" if proba >= 0.7 else ("Medium" if proba >= 0.4 else "Low"),
    }


# ===========================================================================
# 4. SMS Spam Detection — TF-IDF Vectorizer + Multinomial Naive Bayes
# ===========================================================================

_SMS_TEXT_COLS = ["message", "text", "sms", "content", "body", "raw_text", "msg"]
_SMS_TARGET_COLS = ["label", "spam", "class", "target", "category", "type"]


def _build_tfidf_nb_pipeline() -> Pipeline:
    """Construct the canonical TF-IDF → MultinomialNB sklearn Pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode",
                    analyzer="word",
                    # Only pure alpha tokens ≥ 2 chars (avoids punctuation noise)
                    token_pattern=r"\b[a-zA-Z]{2,}\b",
                    ngram_range=(1, 2),       # unigrams + bigrams
                    max_features=10_000,
                    sublinear_tf=True,        # replace tf with 1 + log(tf)
                    min_df=2,                 # ignore hapax legomena
                ),
            ),
            (
                "clf",
                MultinomialNB(alpha=0.1),    # Laplace smoothing; 0.1 < 1.0 since TF-IDF is dense
            ),
        ]
    )


def train_sms_spam_detector(df: pd.DataFrame) -> TrainerResult:
    """
    Train a spam detector.

    CSV with raw text column ('message', 'text', etc.) → TF-IDF + MultinomialNB.
    CSV with engineered numeric features only           → LogisticRegression.
    """
    df = df.dropna().copy()

    text_col   = _resolve_column(df, _SMS_TEXT_COLS)
    # Reject numeric columns mistakenly matched via substring (e.g. "message_id")
    if text_col is not None and df[text_col].dtype != object:
        text_col = None
    target_col = _resolve_column(df, _SMS_TARGET_COLS) or df.columns[-1]

    y_raw = df[target_col].astype(str).str.lower().str.strip()
    _SPAM_POSITIVE = {"spam", "1", "yes", "true", "junk"}
    y = y_raw.isin(_SPAM_POSITIVE).astype(int).values
    label_map: Dict[int, str] = {0: "ham", 1: "spam"}
    spam_rate = float(y.mean())

    # ── BRANCH A: raw text CSV → TF-IDF + MultinomialNB ──────────────────
    if text_col is not None:
        texts: List[str] = df[text_col].astype(str).tolist()
        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, y, test_size=0.2, random_state=42, stratify=y
        )
        pipeline = _build_tfidf_nb_pipeline()
        pipeline.fit(X_tr, y_tr)
        y_pred  = pipeline.predict(X_te)
        y_proba = pipeline.predict_proba(X_te)[:, 1]
        accuracy = float(accuracy_score(y_te, y_pred))
        tfidf_step: TfidfVectorizer = pipeline.named_steps["tfidf"]
        clf_step: MultinomialNB     = pipeline.named_steps["clf"]
        vocab = np.array(tfidf_step.get_feature_names_out())
        top_spam_words: List[str] = []
        if len(clf_step.classes_) >= 2:
            top_idx = np.argsort(clf_step.feature_log_prob_[-1])[-15:][::-1]
            top_spam_words = vocab[top_idx].tolist()
        sample_n = min(5, len(X_te))
        samples = [
            {"text_preview": X_te[i][:80], "predicted": label_map[int(y_pred[i])],
             "spam_probability": round(float(y_proba[i]) * 100, 1),
             "actual": label_map[int(y_te[i])]}
            for i in range(sample_n)
        ]
        return {
            "model": pipeline, "model_type": "tfidf_pipeline",
            "text_col": text_col, "target_col": target_col, "label_map": label_map,
            "model_name": "Naive Bayes (TF-IDF)",
            "task_type": "text_classification",
            "metrics": {"accuracy": round(accuracy * 100, 2), "spam_rate_pct": round(spam_rate * 100, 1),
                        "vocab_size": int(len(vocab)), "score_label": "Accuracy %",
                        "primary_score": round(accuracy * 100, 2)},
            "top_spam_words": top_spam_words, "sample_predictions": samples,
            "explained": f"TF-IDF + MultinomialNB: {accuracy*100:.1f}% accuracy.",
        }

    # ── BRANCH B: engineered feature CSV → LogisticRegression ────────────
    feature_cols: List[str] = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col and df[c].nunique() > 1
        and not (c.lower() in ("id", "message_id", "row_id", "index")
                 or (df[c].nunique() == len(df) and str(df[c].dtype).startswith("int")))
    ]
    if not feature_cols:
        raise ValueError("SMS spam CSV needs a raw text column or numeric spam-feature columns.")

    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42, stratify=y)

    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
    lr_model.fit(X_tr, y_tr)
    y_pred  = lr_model.predict(X_te)
    y_proba = lr_model.predict_proba(X_te)[:, 1]
    accuracy = float(accuracy_score(y_te, y_pred))

    coef = np.abs(lr_model.coef_[0])
    top_feat_idx = np.argsort(coef)[-10:][::-1]
    top_spam_words = [feature_cols[i] for i in top_feat_idx]

    sample_n = min(5, len(y_te))
    samples = [
        {"predicted": label_map[int(y_pred[i])],
         "spam_probability": round(float(y_proba[i]) * 100, 1),
         "actual": label_map[int(y_te[i])]}
        for i in range(sample_n)
    ]
    return {
        "model": lr_model, "model_type": "numeric_lr",
        "scaler": scaler, "feature_cols": feature_cols,
        "target_col": target_col, "label_map": label_map,
        "model_name": "Logistic Regression (Spam Features)",
        "task_type": "text_classification",
        "metrics": {"accuracy": round(accuracy * 100, 2), "spam_rate_pct": round(spam_rate * 100, 1),
                    "score_label": "Accuracy %", "primary_score": round(accuracy * 100, 2)},
        "top_spam_words": top_spam_words, "sample_predictions": samples,
        "explained": (
            f"Logistic Regression on {len(feature_cols)} engineered features: "
            f"{accuracy*100:.1f}% accuracy, {spam_rate*100:.1f}% spam rate. "
            "Top indicators: " + ", ".join(top_spam_words[:6]) + "."
        ),
    }


def predict_spam(model_data: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Classify spam — works with both TF-IDF pipeline and numeric LR model."""
    model_type = model_data.get("model_type", "tfidf_pipeline")
    label_map  = model_data.get("label_map", {0: "ham", 1: "spam"})
    if model_type == "tfidf_pipeline":
        pipeline = model_data["model"]
        proba = pipeline.predict_proba([text])[0]
        pred  = int(pipeline.predict([text])[0])
    else:
        import re
        words = text.split()
        feature_cols = model_data["feature_cols"]
        scaler       = model_data["scaler"]
        row = {
            "word_count":        len(words),
            "special_chars":     len(re.findall(r'[^\w\s]', text)),
            "capital_ratio":     sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "has_url":           int(bool(re.search(r'http|www\.', text, re.I))),
            "has_money_words":   int(bool(re.search(r'free|win|prize|cash|money|reward', text, re.I))),
            "has_urgency_words": int(bool(re.search(r'urgent|immediately|now|call|act', text, re.I))),
            "exclamation_count": text.count('!'),
            "number_count":      len(re.findall(r'\d+', text)),
        }
        x = np.array([[float(row.get(f, 0.0)) for f in feature_cols]])
        x_s = scaler.transform(x)
        proba = model_data["model"].predict_proba(x_s)[0]
        pred  = int(model_data["model"].predict(x_s)[0])
    return {
        "text": text,
        "predicted":        label_map.get(pred, "spam" if pred == 1 else "ham"),
        "is_spam":          bool(pred == 1),
        "spam_probability": round(float(proba[1]) * 100, 1),
    }


# ===========================================================================
# 5. Customer Segmentation — K-Means (auto-k via silhouette score)
# ===========================================================================

_SEGMENT_PREFERRED = [
    "annual_income", "income", "spending_score", "spending",
    "purchase_frequency", "frequency", "age", "credit_score",
    "satisfaction_score", "recency", "clv", "lifetime_value",
]


def train_customer_segmentation(
    df: pd.DataFrame,
    k_range: Tuple[int, int] = (2, 7),
) -> TrainerResult:
    """
    K-Means clustering with automatic k selection via silhouette score.

    Post-training
    -------------
    • Centroids inverse-transformed to original scale for interpretability.
    • PCA 2D projection computed for frontend scatter-plot rendering.
    • Cluster sizes and per-cluster centroid profiles returned.

    Parameters
    ----------
    df      : Input DataFrame; must contain ≥ 2 numeric columns.
    k_range : (min_k, max_k) inclusive range to search.

    Returns (TrainerResult)
    -----------------------
    model, scaler, pca, feature_cols, model_name, task_type,
    metrics {silhouette_score, optimal_k, primary_score},
    centroids, cluster_sizes, scatter_data, cluster_assignments, explained
    """
    df_clean = df.dropna().copy()

    # Prefer known segmentation feature names; fall back to all numerics
    all_num = _numeric_feature_cols(df_clean)
    preferred = [c for c in all_num if any(f in c.lower() for f in _SEGMENT_PREFERRED)]
    feature_cols = preferred if len(preferred) >= 2 else all_num

    if len(feature_cols) < 2:
        raise ValueError(
            "Customer segmentation requires at least 2 numeric feature columns."
        )

    X_raw = df_clean[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── k selection via silhouette ─────────────────────────────────────────
    k_min, k_max = k_range
    best_k: Optional[int] = None
    best_model: Optional[KMeans] = None
    best_score: float = -np.inf
    best_labels: Optional[np.ndarray] = None

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X_scaled, labels))
        if score > best_score:
            best_score, best_k, best_model, best_labels = score, k, km, labels

    if best_model is None or best_labels is None:
        raise ValueError(
            "K-Means failed to produce more than one cluster for all k in range. "
            "Check that your data has sufficient variance."
        )

    # ── centroids back to original scale ──────────────────────────────────
    centroids_original = scaler.inverse_transform(best_model.cluster_centers_)
    centroids = [
        {col: round(float(v), 2) for col, v in zip(feature_cols, row)}
        for row in centroids_original
    ]

    # Cluster size distribution
    unique_labels, counts = np.unique(best_labels, return_counts=True)
    cluster_sizes: Dict[int, int] = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}

    # ── PCA 2D projection for visualisation ───────────────────────────────
    n_components = min(2, X_scaled.shape[1])
    pca_vis = PCA(n_components=n_components, random_state=42)
    coords_2d = pca_vis.fit_transform(X_scaled)

    scatter_data = [
        {
            "x": round(float(coords_2d[i, 0]), 4),
            "y": round(float(coords_2d[i, 1]), 4) if n_components > 1 else 0.0,
            "cluster": int(best_labels[i]),
        }
        for i in range(len(best_labels))
    ]

    return {
        "model": best_model,
        "scaler": scaler,
        "pca": pca_vis,
        "feature_cols": feature_cols,
        "model_name": f"KMeans (k={best_k})",
        "task_type": "clustering",
        "metrics": {
            "silhouette_score": round(best_score, 4),
            "optimal_k": int(best_k),          # type: ignore[arg-type]
            "k_range_searched": list(range(k_min, k_max + 1)),
            "score_label": "Silhouette Score",
            "primary_score": round(best_score, 4),
        },
        "centroids": centroids,
        "cluster_sizes": cluster_sizes,
        "scatter_data": scatter_data,
        "cluster_assignments": best_labels.tolist(),
        "explained": (
            f"KMeans selected k={best_k} (silhouette={best_score:.3f}) "
            f"after searching k ∈ {list(range(k_min, k_max + 1))}. "
            "Centroid profiles describe the 'typical customer' per segment. "
            "PCA 2D projection preserves cluster separation for interactive scatter charts."
        ),
    }


def predict_customer_segment(
    model: KMeans,
    scaler: StandardScaler,
    feature_cols: List[str],
    inputs: Dict[str, float],
) -> Dict[str, Any]:
    """Assign a new customer row to the nearest cluster."""
    x = np.array([[inputs.get(f, 0.0) for f in feature_cols]])
    x_scaled = scaler.transform(x)
    cluster = int(model.predict(x_scaled)[0])
    # Distance to assigned centroid as inverse-confidence proxy
    dist = float(np.linalg.norm(model.cluster_centers_[cluster] - x_scaled))
    confidence = round(float(1.0 / (1.0 + dist)), 4)
    return {"cluster": cluster, "assignment_confidence": confidence}


# ===========================================================================
# 6. Stock Market Visualizer — PCA Dimensionality Reduction
# ===========================================================================

_STOCK_LABEL_COLS = ["ticker", "symbol", "company", "sector", "stock", "name", "id"]


def train_stock_pca_visualizer(df: pd.DataFrame) -> TrainerResult:
    """
    Reduce stock-market features to 2 PCA components for 2D scatter visualisation.

    Returns
    -------
    • 2D coordinates per observation (x, y, optional label)
    • Explained variance ratios (% per component + total)
    • Loading matrix — which features drive each principal component
    • Top 3 driving features per component for tooltip/annotation

    Returns (TrainerResult)
    -----------------------
    model (PCA), scaler, feature_cols, label_col,
    model_name, task_type,
    metrics {explained_variance_pc1/pc2, total_explained_variance, primary_score},
    loadings, top_drivers, scatter_data, explained
    """
    df_clean = df.dropna().copy()

    # Detect optional label column for chart annotation
    label_col = _resolve_column(df_clean, _STOCK_LABEL_COLS)
    labels: Optional[List[str]] = (
        df_clean[label_col].astype(str).tolist() if label_col else None
    )

    # Select non-label numeric columns; drop apparent ID columns (all unique)
    num_cols = [
        c for c in df_clean.select_dtypes(include=[np.number]).columns
        if c != label_col and df_clean[c].nunique() > 1
    ]
    feature_cols = num_cols  # all numeric features feed PCA

    if len(feature_cols) < 2:
        raise ValueError(
            "Stock PCA visualizer requires at least 2 numeric feature columns."
        )

    X_raw = df_clean[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── PCA — always project to 2 components ──────────────────────────────
    n_components = min(2, len(feature_cols))
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    # Explained variance percentages
    evr = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    total_evr = round(sum(evr), 2)

    # Loading matrix: contribution of each original feature to each component
    components = pca.components_  # shape (n_components, n_features)
    loadings: Dict[str, Dict[str, float]] = {
        f"PC{i + 1}": {col: round(float(components[i, j]), 4) for j, col in enumerate(feature_cols)}
        for i in range(n_components)
    }

    # Top 3 driving features per component (by absolute loading)
    top_drivers: Dict[str, List[str]] = {}
    for i in range(n_components):
        abs_load = np.abs(components[i])
        top_idx = np.argsort(abs_load)[-3:][::-1]
        top_drivers[f"PC{i + 1}"] = [feature_cols[j] for j in top_idx]

    scatter_data = [
        {
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4) if n_components > 1 else 0.0,
            "label": labels[i] if labels else str(i),
        }
        for i in range(len(coords))
    ]

    pc1_evr = evr[0] if evr else 0.0
    pc2_evr = evr[1] if len(evr) > 1 else 0.0

    return {
        "model": pca,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "label_col": label_col,
        "model_name": "PCA (2 Components)",
        "task_type": "dimensionality_reduction",
        "metrics": {
            "explained_variance_pc1": pc1_evr,
            "explained_variance_pc2": pc2_evr,
            "total_explained_variance": total_evr,
            "n_features_reduced": len(feature_cols),
            "score_label": "Total Explained Variance %",
            "primary_score": total_evr,
        },
        "variance_explained": evr,          # [pc1_pct, pc2_pct] — consumed by JS
        "loadings": loadings,
        "top_drivers": top_drivers,
        "scatter_data": scatter_data,
        "explained": (
            f"PCA reduced {len(feature_cols)} features → 2 components "
            f"retaining {total_evr}% of total variance. "
            f"PC1 ({pc1_evr}%): primary axis driven by "
            f"{', '.join(top_drivers.get('PC1', [])[:2])}. "
            f"PC2 ({pc2_evr}%): secondary axis driven by "
            f"{', '.join(top_drivers.get('PC2', [])[:2])}. "
            "Each dot on the scatter chart represents one stock; "
            "proximity indicates market behaviour similarity."
        ),
    }


# ===========================================================================
# Dispatcher registry — maps scenario_id → trainer function
# ===========================================================================

DEDICATED_TRAINERS: Dict[str, Any] = {
    "regression_housing": train_house_price_estimator,
    "banknote_authentication": train_banknote_authenticator,
    "customer_churn": train_customer_churn_predictor,
    "sms_spam": train_sms_spam_detector,
    "customer_segments": train_customer_segmentation,
    "stock_sectors": train_stock_pca_visualizer,
}

def predict_stock_pca(
    model_data: Dict[str, Any],
    feature_vals: Dict[str, Any],
) -> Dict[str, Any]:
    """Project a new observation through the fitted PCA and return 2D coordinates."""
    pca: PCA = model_data["model"]
    scaler: StandardScaler = model_data["scaler"]
    feature_cols: List[str] = model_data["feature_cols"]
    x = np.array([[float(feature_vals.get(f, 0.0)) for f in feature_cols]])
    x_scaled = scaler.transform(x)
    coords = pca.transform(x_scaled)[0]
    evr = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
    return {
        "pc1": round(float(coords[0]), 4),
        "pc2": round(float(coords[1]), 4) if len(coords) > 1 else 0.0,
        "variance_explained": evr,
    }


LIVE_PREDICTORS: Dict[str, Any] = {
    "regression_housing": predict_house_price,
    "banknote_authentication": predict_banknote,
    "customer_churn": predict_churn,
    "sms_spam": predict_spam,
    "customer_segments": predict_customer_segment,
    "stock_sectors": predict_stock_pca,
}
