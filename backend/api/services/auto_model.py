"""
Enhanced AutoML Service with All 10 ML Models and Real-World Explanations

Automatically detects task type, runs model tournament, and provides
business-friendly insights with real-world analogies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from .model_explanations import get_model_explanation


def recommend_and_run_best_model(
    df: pd.DataFrame,
    target_col: Optional[str],
    user_intent: str = "",
    business_objective: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run AutoML with expanded model library and return results with real-world explanations.
    
    Supports all 10 requested models:
    - Linear Regression
    - Logistic Regression
    - Decision Tree
    - KNN
    - SVM
    - Random Forest
    - K-Means Clustering
    - Naive Bayes
    - PCA (Dimensionality Reduction)
    - XGBoost
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty.")
    
    # Basic cleaning
    df = df.dropna()
    if df.empty:
        raise ValueError("All rows dropped after removing NaNs; please provide cleaner data.")

    # Detect task type
    task_type = detect_task_type(df, target_col, business_objective)

    # Handle unsupervised tasks
    if task_type == "clustering":
        best_name, best_model, best_score, cluster_labels = run_clustering_tournament(df)
        reasoning = f"Selected because it achieved the highest silhouette score of {best_score:.3f} across tested k."
        summary = summarize_business_insights(
            task_type=task_type,
            predictions=cluster_labels,
            source_df=df,
            target_col=target_col,
            business_objective=business_objective,
        )
        explanation = get_model_explanation(best_name, task_type)
        
        return {
            "recommended_model_name": best_name,
            "reasoning": reasoning,
            "model_object": best_model,
            "task_type": task_type,
            "metric_value": best_score,
            "business_summary": summary,
            "model_explanation": explanation,
        }

    if task_type == "anomaly":
        best_name, best_model, best_score, anomaly_flags = run_anomaly_tournament(df)
        reasoning = "Selected Isolation Forest for its balance of precision and coverage on outliers."
        summary = summarize_business_insights(
            task_type=task_type,
            predictions=anomaly_flags,
            source_df=df,
            target_col=None,
            business_objective=business_objective,
        )
        explanation = get_model_explanation(best_name, task_type)
        
        return {
            "recommended_model_name": best_name,
            "reasoning": reasoning,
            "model_object": best_model,
            "task_type": task_type,
            "metric_value": best_score,
            "business_summary": summary,
            "model_explanation": explanation,
        }

    # Supervised learning path
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Build preprocessing pipeline
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
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=10),
            "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            candidates["XGBoost Regressor"] = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        
        scorer = r2_score
        best_name, best_model, best_score = run_supervised_tournament(
            candidates, preprocessor, X_train, X_test, y_train, y_test, scorer
        )
        reasoning = f"Selected because it achieved the highest R² of {best_score:.3f} among tested regressors."
        
    else:  # classification
        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, max_depth=10),
            "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
            "SVM Classifier": SVC(kernel='rbf', random_state=42, probability=True),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
            "Naive Bayes": GaussianNB(),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            candidates["XGBoost Classifier"] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        scorer = accuracy_score
        best_name, best_model, best_score = run_supervised_tournament(
            candidates, preprocessor, X_train, X_test, y_train, y_test, scorer
        )
        reasoning = f"Selected because it achieved the highest accuracy of {best_score:.3f} among tested classifiers."

    # Refit on full dataset
    best_model.fit(X, y)
    predictions = best_model.predict(X)

    summary = summarize_business_insights(
        task_type=task_type,
        predictions=predictions,
        source_df=df,
        target_col=target_col,
        business_objective=business_objective,
    )
    
    # Get real-world explanation
    explanation = get_model_explanation(best_name, task_type)

    return {
        "recommended_model_name": best_name,
        "reasoning": reasoning,
        "model_object": best_model,
        "task_type": task_type,
        "metric_value": best_score,
        "business_summary": summary,
        "model_explanation": explanation,
    }


def detect_task_type(
    df: pd.DataFrame,
    target_col: Optional[str],
    business_objective: Optional[str] = None
) -> str:
    """Detect whether task is classification, regression, clustering, or anomaly detection."""
    objective = (business_objective or "").lower()
    
    if "anomaly" in objective or "fraud" in objective or "outlier" in objective:
        return "anomaly"
    if "segment" in objective or "cluster" in objective:
        return "clustering"
    if target_col is None:
        return "clustering"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    series = df[target_col]
    if pd.api.types.is_numeric_dtype(series):
        return "regression" if series.nunique() > 20 or "revenue" in objective else "classification"
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
    """Run tournament among supervised learning models."""
    best_name, best_model, best_score = None, None, -np.inf
    
    for name, estimator in candidates.items():
        try:
            model = Pipeline([("prep", preprocessor), ("model", estimator)]) if preprocessor else estimator
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = scorer(y_test, preds)
            
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model
        except Exception as e:
            # Skip models that fail (e.g., Naive Bayes with negative features)
            print(f"Model {name} failed: {e}")
            continue
    
    if best_model is None:
        raise RuntimeError("All models failed during tournament. Check your data.")
    
    return best_name, best_model, float(best_score)


def run_clustering_tournament(df: pd.DataFrame) -> Tuple[str, Any, float, np.ndarray]:
    """Run tournament for clustering algorithms."""
    X = df.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("Clustering requires at least one numeric column.")
    
    # Apply PCA for dimensionality reduction if needed
    n_components = min(5, X.shape[1])
    pca = PCA(n_components=n_components) if n_components >= 3 else None

    best_score = -np.inf
    best_k = None
    best_model = None
    best_labels = None

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
                best_labels = labels
                
    if best_model is None or best_labels is None:
        raise ValueError("Clustering failed to produce more than one cluster; please check your data.")
    
    return f"KMeans (k={best_k})", best_model, float(best_score), np.asarray(best_labels)


def run_anomaly_tournament(df: pd.DataFrame) -> Tuple[str, Any, float, np.ndarray]:
    """Run anomaly detection using Isolation Forest."""
    X = df.select_dtypes(include=[np.number])
    if X.empty:
        raise ValueError("Anomaly detection requires at least one numeric column.")
    
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("iso", IsolationForest(random_state=42, contamination=0.1)),
        ]
    )
    flags = pipe.fit_predict(X)  # -1 = anomaly, 1 = normal
    anomaly_rate = float((flags == -1).mean())
    score = 1 - anomaly_rate  # higher is better (fewer anomalies)
    
    return "Isolation Forest", pipe, score, np.asarray(flags)


def summarize_business_insights(
    task_type: str,
    predictions: np.ndarray,
    source_df: pd.DataFrame,
    target_col: Optional[str],
    business_objective: Optional[str],
) -> Dict[str, Any]:
    """Generate business-friendly summary of predictions."""
    objective = (business_objective or "").lower()
    total = len(source_df)
    summary: Dict[str, Any] = {"headline": "", "details": {}, "recommended_action": "", "detailed_insight": ""}

    if task_type == "classification":
        # Try to identify "positive" class
        positive_mask = np.isin(
            np.array([str(p).lower() for p in predictions]),
            ["1", "true", "yes", "churn", "at_risk", "risk", "high"],
        )
        at_risk = int(positive_mask.sum())
        pct = (at_risk / total) * 100 if total else 0
        summary["headline"] = f"ALERT: {at_risk} entities flagged as high risk ({pct:.1f}% of records)."
        summary["details"] = {"high_risk_count": at_risk, "high_risk_percent": round(pct, 1)}
        summary["recommended_action"] = "Prioritize outreach to high-risk customers and launch retention offers."
        summary["detailed_insight"] = f"Out of {total} records analyzed, {at_risk} were classified as high-risk based on the patterns detected."
        return summary

    if task_type == "regression":
        preds = np.asarray(predictions, dtype=float)
        avg_value = float(np.mean(preds))
        top_decile_cut = float(np.percentile(preds, 90))
        top_decile_count = int((preds >= top_decile_cut).sum())
        summary["headline"] = f"OPPORTUNITY: Top decile threshold is {top_decile_cut:.2f}. Avg prediction {avg_value:.2f}."
        summary["details"] = {
            "average_prediction": round(avg_value, 2),
            "top_decile_threshold": round(top_decile_cut, 2),
            "top_decile_count": top_decile_count,
        }
        summary["recommended_action"] = "Target the top-decile segment with premium upsell campaigns."
        summary["detailed_insight"] = f"The average predicted value is {avg_value:.2f}. Focus on the top {top_decile_count} records (top 10%) with values above {top_decile_cut:.2f}."
        return summary

    if task_type == "clustering":
        labels = np.asarray(predictions)
        counts = {int(lbl): int((labels == lbl).sum()) for lbl in np.unique(labels)}
        dominant_cluster = max(counts, key=counts.get)
        share = (counts[dominant_cluster] / total) * 100 if total else 0
        summary["headline"] = f"SEGMENTATION: Found {len(counts)} clusters. Cluster {dominant_cluster} is {share:.1f}% of the base."
        summary["details"] = {"clusters": counts, "dominant_cluster": dominant_cluster, "dominant_share_percent": round(share, 1)}
        summary["recommended_action"] = "Design tailored messaging per segment; start with the dominant cluster to maximize impact."
        summary["detailed_insight"] = f"Your data naturally segments into {len(counts)} distinct groups. The largest group (Cluster {dominant_cluster}) contains {counts[dominant_cluster]} records."
        return summary

    if task_type == "anomaly":
        flags = np.asarray(predictions)
        anomalies = int((flags == -1).sum())
        pct = (anomalies / total) * 100 if total else 0
        summary["headline"] = f"ANOMALIES: {anomalies} potential outliers detected ({pct:.1f}% of records)."
        summary["details"] = {"anomaly_count": anomalies, "anomaly_percent": round(pct, 1)}
        summary["recommended_action"] = "Investigate flagged records for fraud or data quality issues."
        summary["detailed_insight"] = f"Out of {total} records, {anomalies} appear unusual based on statistical patterns. These warrant further investigation."
        return summary

    summary["headline"] = "Analysis complete."
    summary["recommended_action"] = "Review dataset and objective."
    return summary
