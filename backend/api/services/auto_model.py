from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor


def recommend_and_run_best_model(
    df: pd.DataFrame,
    target_col: Optional[str],
    user_intent: str = "",
) -> Dict[str, Any]:
    """
    Auto-detect task type, run a small tournament, and return the top model.

    Drops NaNs defensively. Returns a dictionary with:
        recommended_model_name, reasoning, model_object, task_type, metric_value.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")
    df = df.dropna()
    if df.empty:
        raise ValueError("All rows dropped after removing NaNs; please provide cleaner data.")

    task_type = detect_task_type(df, target_col)

    if task_type == "clustering":
        best_name, best_model, best_score = run_clustering_tournament(df)
        reasoning = f"Selected because it achieved the highest silhouette score of {best_score:.3f} across tested k."
        return {
            "recommended_model_name": best_name,
            "reasoning": reasoning,
            "model_object": best_model,
            "task_type": task_type,
            "metric_value": best_score,
        }

    # Supervised path
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers) if transformers else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if task_type == "classification" else None,
    )

    if task_type == "regression":
        candidates = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
        }
        scorer = r2_score
        best_name, best_model, best_score = run_supervised_tournament(
            candidates, preprocessor, X_train, X_test, y_train, y_test, scorer
        )
        reasoning = f"Selected because it achieved the highest R² of {best_score:.3f} among tested regressors."
    else:  # classification
        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "SVC": SVC(),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42),
        }
        scorer = accuracy_score
        best_name, best_model, best_score = run_supervised_tournament(
            candidates, preprocessor, X_train, X_test, y_train, y_test, scorer
        )
        reasoning = f"Selected because it achieved the highest accuracy of {best_score:.3f} among tested classifiers."

    # Refit best model on all data for downstream predictions
    best_model.fit(X, y)

    return {
        "recommended_model_name": best_name,
        "reasoning": reasoning,
        "model_object": best_model,
        "task_type": task_type,
        "metric_value": best_score,
    }


def detect_task_type(df: pd.DataFrame, target_col: Optional[str]) -> str:
    if target_col is None:
        return "clustering"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    series = df[target_col]
    if pd.api.types.is_numeric_dtype(series):
        return "regression" if series.nunique() > 20 else "classification"
    return "classification"


def run_supervised_tournament(
    candidates: Dict[str, Any],
    preprocessor: Optional[ColumnTransformer],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scorer,
) -> Tuple[str, Any, float]:
    best_name, best_model, best_score = None, None, -np.inf
    for name, estimator in candidates.items():
        model = Pipeline([("prep", preprocessor), ("model", estimator)]) if preprocessor else estimator
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = scorer(y_test, preds)
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
    return best_name, best_model, float(best_score)


def run_clustering_tournament(df: pd.DataFrame) -> Tuple[str, Any, float]:
    X = df.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("Clustering requires at least one numeric column.")
    n_components = min(5, X.shape[1])
    pca = PCA(n_components=n_components) if n_components >= 3 else None

    best_score = -np.inf
    best_k = None
    best_model = None

    for k in [3, 4, 5]:
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        if pca:
            steps.append(("pca", pca))
        steps.append(("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)))
        pipe = Pipeline(steps)
        labels = pipe.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_model = pipe
    if best_model is None:
        raise ValueError("Clustering failed to produce more than one cluster; please check your data.")
    return f"KMeans (k={best_k})", best_model, float(best_score)
