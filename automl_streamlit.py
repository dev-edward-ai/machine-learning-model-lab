"""Unified AutoML logic and Streamlit UI.

This file exposes recommend_and_run_best_model(df, target_col, user_intent)
plus a simple Streamlit interface for CSV upload and intent capture.
"""

from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_cyberpunk_design() -> None:
    """Inject cyberpunk/dark-mode CSS for Streamlit UI."""
    st.markdown(
        """
        <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@400;600&display=swap');

        /* Global background and text */
        html, body, [class^="css"]  {
            background: #0E1117;
            color: #FAFAFA;
            font-family: 'Inter', 'Roboto', sans-serif;
        }

        /* Headers use Orbitron for sci-fi feel */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Orbitron', 'Inter', sans-serif;
            letter-spacing: 0.03em;
            color: #FAFAFA;
        }

        /* Buttons: neon border, glow on hover */
        div.stButton > button {
            background: linear-gradient(135deg, #0E1117 0%, #0E1117 100%);
            color: #FAFAFA;
            border: 1px solid #00FF99;
            border-radius: 6px;
            box-shadow: 0 0 10px rgba(0, 255, 153, 0.35);
            transition: all 0.2s ease;
        }
        div.stButton > button:hover {
            border-color: #00C9FF;
            box-shadow: 0 0 18px rgba(0, 201, 255, 0.55), 0 0 8px rgba(0, 255, 153, 0.45);
            transform: translateY(-1px);
        }

        /* Inputs and file uploader: dark background with neon focus */
        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stDateInput input,
        .stSelectbox > div > div > select,
        .stNumberInput input,
        .stFileUploader div[data-testid="stFileUploadDropzone"] {
            background: #262730;
            color: #FAFAFA;
            border-radius: 6px;
            border: 1px solid #2e2f36;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea textarea:focus,
        .stDateInput input:focus,
        .stSelectbox > div > div > select:focus,
        .stNumberInput input:focus,
        .stFileUploader div[data-testid="stFileUploadDropzone"]:focus {
            outline: none;
            border-color: #00C9FF;
            box-shadow: 0 0 10px rgba(0, 201, 255, 0.4);
        }

        /* File uploader text color */
        .stFileUploader label div {
            color: #FAFAFA;
        }

        /* Glassmorphism cards for info/success/warning/error/metric */
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #00FF99;
        }
        [data-testid="stAlert"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 201, 255, 0.35);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(8px);
            border-radius: 10px;
            color: #FAFAFA;
        }
        /* Generic container tweaks */
        .block-container {
            padding-top: 1.5rem;
        }
        section.main > div {
            background: #0E1117;
        }

        /* Hide default Streamlit header, footer, and hamburger */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode object/category columns in-place and return the frame."""
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object" or str(df_encoded[col].dtype) == "category":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded


def recommend_and_run_best_model(
    df: pd.DataFrame,
    target_col: t.Optional[str],
    user_intent: str,
) -> dict:
    """
    Auto-detect task, run a small model tournament, and return the winner with reasoning.

    Returns a dict with: task, model_name, model_object, score, metric, reasoning.
    """

    if df is None or df.empty:
        raise ValueError("DataFrame is empty. Please provide data.")

    df_work = df.copy()

    if target_col and target_col not in df_work.columns:
        raise ValueError(f"Target column '{target_col}' not found in the data.")

    # Basic cleaning for stability.
    df_work = df_work.dropna()
    if df_work.empty:
        raise ValueError("All rows were dropped due to NaN values. Provide cleaner data.")

    # Task detection.
    if target_col is None:
        task = "clustering"
    else:
        target_series = df_work[target_col]
        unique_vals = target_series.nunique()
        is_numeric_target = pd.api.types.is_numeric_dtype(target_series)
        if is_numeric_target and unique_vals > 20:
            task = "regression"
        else:
            task = "classification"

    results = []

    if task == "classification":
        X = df_work.drop(columns=[target_col])
        y_raw = df_work[target_col]

        X_enc = _encode_categoricals(X)
        y_enc = LabelEncoder().fit_transform(y_raw.astype(str))

        stratify = y_enc if len(np.unique(y_enc)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y_enc, test_size=0.2, random_state=42, stratify=stratify
        )

        candidates = [
            ("Logistic Regression", LogisticRegression(max_iter=500, n_jobs=None)),
            ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        ]
        for name, model in candidates:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)
            results.append({
                "model_name": name,
                "model_object": model,
                "score": float(score),
                "metric": "accuracy",
            })

    elif task == "regression":
        X = df_work.drop(columns=[target_col])
        y = df_work[target_col].astype(float)

        X_enc = _encode_categoricals(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=0.2, random_state=42
        )

        candidates = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42)),
            ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
        ]
        for name, model in candidates:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds)
            results.append({
                "model_name": name,
                "model_object": model,
                "score": float(score),
                "metric": "r2",
            })

    else:  # clustering
        X = _encode_categoricals(df_work)
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        candidates = []
        for k in [3, 5]:
            candidates.append((f"KMeans (k={k})", KMeans(n_clusters=k, random_state=42)))

        for name, model in candidates:
            model.fit(X_train)
            try:
                labels = model.predict(X_test)
                if len(np.unique(labels)) < 2:
                    raise ValueError("Silhouette undefined for a single cluster.")
                score = silhouette_score(X_test, labels)
            except Exception:
                score = float("-inf")
            results.append({
                "model_name": name,
                "model_object": model,
                "score": float(score),
                "metric": "silhouette",
            })

    if not results:
        raise RuntimeError("No models were evaluated.")

    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    best = results_sorted[0]

    metric = best["metric"]
    score_pct = best["score"] * 100 if metric == "accuracy" else best["score"]
    score_str = f"{score_pct:.2f}%" if metric == "accuracy" else f"{score_pct:.3f}"

    reasoning = (
        f"Based on your goal to '{user_intent}', we ran a competitive analysis for {task}. "
        f"{best['model_name']} led with {score_str} {metric}, delivering a stronger fit "
        f"than alternative approaches for the patterns present in your data."
    )

    return {
        "task": task,
        "model_name": best["model_name"],
        "model_object": best["model_object"],
        "score": best["score"],
        "metric": metric,
        "reasoning": reasoning,
    }


# -------- Streamlit UI (same file by request) --------
st.set_page_config(page_title="AutoML Insight", page_icon="🤖", layout="wide")
load_cyberpunk_design()
st.title("AutoML Logic Layer (Consultant Mode)")

st.markdown(
    "Upload a CSV, describe your goal, and let the built-in tournament pick the best model."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
user_intent = st.text_input("Goal / Intent", placeholder="e.g., Find sick patients")

if uploaded is not None:
    df_uploaded = pd.read_csv(uploaded)
    st.write("Preview", df_uploaded.head())
    target_options = ["Unsupervised (no target)"] + list(df_uploaded.columns)
    target_choice = st.selectbox("Target column (choose 'Unsupervised' for clustering)", target_options)
    target_selected = None if target_choice == "Unsupervised (no target)" else target_choice
else:
    df_uploaded = None
    target_selected = None

run_btn = st.button("Run AutoML")

if run_btn:
    if df_uploaded is None:
        st.error("Please upload a CSV first.")
    elif not user_intent.strip():
        st.error("Please enter your goal/intent to contextualize the analysis.")
    else:
        with st.spinner("Running tournament..."):
            try:
                outcome = recommend_and_run_best_model(
                    df=df_uploaded,
                    target_col=target_selected,
                    user_intent=user_intent.strip(),
                )
                st.success(outcome["reasoning"])
                st.metric(
                    label=f"Best model ({outcome['metric']})",
                    value=f"{outcome['model_name']} | {outcome['score']:.3f}",
                )
                st.write("Model details:", outcome["model_object"])
            except Exception as exc:  # surface friendly errors
                st.error(f"Analysis failed: {exc}")
