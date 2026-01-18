from __future__ import annotations

import io
import typing as t

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
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
    if df is None or df.empty:
        raise ValueError("DataFrame is empty. Please provide data.")

    df_work = df.copy()

    if target_col and target_col not in df_work.columns:
        raise ValueError(f"Target column '{target_col}' not found in the data.")

    df_work = df_work.dropna()
    if df_work.empty:
        raise ValueError("All rows were dropped due to NaN values. Provide cleaner data.")

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


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported."}), 400

        content = file.stream.read()
        df = pd.read_csv(io.BytesIO(content))

        target_col = request.form.get("target_col") or None
        user_intent = request.form.get("user_intent") or "Your stated goal"

        outcome = recommend_and_run_best_model(
            df=df, target_col=target_col, user_intent=user_intent
        )

        preview = df.head(5).to_dict(orient="records")

        return jsonify({
            "recommended_model": outcome["model_name"],
            "score": outcome["score"],
            "reasoning": outcome["reasoning"],
            "preview_data": preview,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
