"""
Smart Dispatcher System

Automatically tests uploaded CSV against multiple models and returns:
- Performance metrics for top models
- Best scenario match
- Confidence scores
- Real-world recommendations
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

from .auto_model import (
    detect_task_type,
    run_supervised_tournament,
    run_clustering_tournament,
    run_anomaly_tournament
)
from .model_explanations import get_model_explanation
from .model_cache import get_model_cache


# Real-world scenario definitions
SCENARIOS = {
    "crypto_signals": {
        "name": "Crypto Buy/Sell Signal",
        "description": "Binary classification for cryptocurrency trading signals",
        "model_type": "Logistic Regression",
        "task": "classification",
        "icon": "💰",
        "keywords": ["price", "signal", "rsi", "moving", "macd", "buy", "sell"],
        "industry": "Finance/Trading"
    },
    "loan_applications": {
        "name": "Loan Approval Assistant",
        "description": "Interpretable loan approval decisions",
        "model_type": "Decision Tree Classifier",
        "task": "classification",
        "icon": "🏦",
        "keywords": ["income", "credit", "loan", "approved", "debt", "employment"],
        "industry": "Banking/Finance"
    },
    "sms_spam": {
        "name": "Spam vs Ham SMS Detector",
        "description": "Text-based spam filtering",
        "model_type": "Naive Bayes",
        "task": "classification",
        "icon": "📱",
        "keywords": ["word", "message", "special_chars", "spam", "url", "urgency"],
        "industry": "Communications"
    },
    "banknote_authentication": {
        "name": "Fake Banknote Detector",
        "description": "Precision boundary classification for counterfeit detection",
        "model_type": "SVM Classifier",
        "task": "classification",
        "icon": "💵",
        "keywords": ["variance", "skewness", "curtosis", "entropy", "authentic"],
        "industry": "Security/Banking"
    },
    "heart_disease": {
        "name": "Disease Risk Predictor",
        "description": "Medical ensemble classification",
        "model_type": "Random Forest Classifier",
        "task": "classification",
        "icon": "❤️",
        "keywords": ["age", "cholesterol", "heart", "disease", "blood_pressure", "ecg"],
        "industry": "Healthcare"
    },
    "customer_churn": {
        "name": "Customer Churn Predictor",
        "description": "Predict customer subscription cancellation",
        "model_type": "XGBoost Classifier",
        "task": "classification",
        "icon": "📊",
        "keywords": ["tenure", "monthly", "churn", "contract", "charges", "customer"],
        "industry": "SaaS/Subscription"
    },
    "marketing_roi": {
        "name": "Marketing Ad ROI Calculator",
        "description": "Linear trend prediction for advertising ROI",
        "model_type": "Linear Regression",
        "task": "regression",
        "icon": "📈",
        "keywords": ["ad_spend", "sales", "impressions", "clicks", "campaign", "roi"],
        "industry": "Marketing/Advertising"
    },
    "used_car_prices": {
        "name": "Used Car Price Estimator",
        "description": "Non-linear pricing for vehicles",
        "model_type": "Decision Tree Regressor",
        "task": "regression",
        "icon": "🚗",
        "keywords": ["price", "mileage", "year", "brand", "engine", "warranty"],
        "industry": "Automotive/E-commerce"
    },
    "airbnb_pricing": {
        "name": "Airbnb Nightly Rate Estimator",
        "description": "Neighborhood-based pricing",
        "model_type": "KNN Regressor",
        "task": "regression",
        "icon": "🏠",
        "keywords": ["nightly_rate", "neighborhood", "accommodates", "wifi", "pool", "review"],
        "industry": "Hospitality/Real Estate"
    },
    "flight_delays": {
        "name": "Flight Delay Prediction",
        "description": "Complex interactions for delay forecasting",
        "model_type": "Random Forest Regressor",
        "task": "regression",
        "icon": "✈️",
        "keywords": ["delay", "airline", "weather", "traffic", "flight", "departure"],
        "industry": "Aviation/Travel"
    },
    "color_palette": {
        "name": "Image Color Palette Generator",
        "description": "Pixel grouping for dominant colors",
        "model_type": "KMeans",
        "task": "clustering",
        "icon": "🎨",
        "keywords": ["red", "green", "blue", "rgb", "pixel", "color"],
        "industry": "Design/Graphics"
    },
    "stock_sectors": {
        "name": "Stock Market Sector Visualizer",
        "description": "Dimensionality reduction for visualization",
        "model_type": "PCA",
        "task": "dimensionality_reduction",
        "icon": "📉",
        "keywords": ["stock", "sector", "tech_score", "market", "beta", "volatility"],
        "industry": "Finance/Investment"
    },
    "credit_card_transactions": {
        "name": "Credit Card Fraud Detection",
        "description": "Anomaly detection for suspicious transactions",
        "model_type": "Isolation Forest",
        "task": "anomaly",
        "icon": "🔍",
        "keywords": ["transaction", "amount", "fraud", "merchant", "distance", "online"],
        "industry": "Banking/Security"
    }
}


def _scenario_model_matches(result_name: str, scenario_model_type: str) -> bool:
    """Check if a tournament result name matches the scenario's designated model."""
    if not scenario_model_type or not result_name:
        return False
    a = result_name.lower().strip()
    b = scenario_model_type.lower().strip()
    return b in a or a.startswith(b) or (b.startswith("kmeans") and "kmeans" in a)


def _find_scenario_model_index(
    results: List[Tuple[str, float, Any]], scenario_model_type: str
) -> Optional[int]:
    """Return index of first result whose name matches the scenario model, else None."""
    for i, (name, _, _) in enumerate(results):
        if _scenario_model_matches(name, scenario_model_type):
            return i
    return None


def detect_scenario(df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[str, float]:
    """
    Detect which real-world scenario the dataset matches.
    Returns (scenario_id, confidence_score)
    """
    columns = [col.lower() for col in df.columns]
    best_match = None
    best_score = 0.0
    
    for scenario_id, scenario in SCENARIOS.items():
        keywords = scenario["keywords"]
        matches = sum(1 for keyword in keywords if any(keyword in col for col in columns))
        score = matches / len(keywords) if keywords else 0.0
        
        if score > best_score:
            best_score = score
            best_match = scenario_id
    
    # Default to generic if no strong match
    if best_score < 0.2:
        best_match = "general_analysis"
        best_score = 0.5
    
    return best_match, best_score


def smart_dispatch(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    business_objective: Optional[str] = None
) -> Dict[str, Any]:
    """
    Smart Dispatcher - Tests models and returns comprehensive analysis.
    
    Returns:
        {
            "scenario": {...},
            "top_models": [{name, score, model_type, explanation}, ...],
            "recommended_model": {...},
            "dataset_summary": {...},
            "confidence": float
        }
    """
    
    # Detect scenario
    scenario_id, scenario_confidence = detect_scenario(df, target_col)
    scenario = SCENARIOS.get(scenario_id, {
        "name": "General Data Analysis",
        "description": "Custom analysis for your dataset",
        "icon": "🤖",
        "task": "unknown",
        "industry": "General"
    })
    
    # Detect task type
    task_type = detect_task_type(df, target_col, business_objective)
    
    # Dataset summary
    dataset_summary = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "num_numeric": len(df.select_dtypes(include=[np.number]).columns),
        "num_categorical": len(df.select_dtypes(exclude=[np.number]).columns),
        "missing_values": int(df.isnull().sum().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
    
    # Run model tournament based on task type
    winning_model = None
    winning_preprocessor = None
    feature_names = []
    
    if task_type == "clustering":
        best_name, best_model, best_score, _ = run_clustering_tournament(df)
        winning_model = best_model
        feature_names = df.columns.tolist()
        top_models = [{
            "name": best_name,
            "score": round(float(best_score), 3),
            "score_type": "Silhouette Score",
            "model_type": "clustering",
            "explanation": get_model_explanation(best_name, task_type)
        }]
        
    elif task_type == "anomaly":
        best_name, best_model, best_score, _ = run_anomaly_tournament(df)
        winning_model = best_model
        feature_names = df.columns.tolist()
        top_models = [{
            "name": best_name,
            "score": round(float(best_score), 3),
            "score_type": "Normal Data %",
            "model_type": "anomaly_detection",
            "explanation": get_model_explanation(best_name, task_type)
        }]
        
    else:
        # Supervised learning - run tournament and get top 3
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.naive_bayes import GaussianNB
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        
        try:
            from xgboost import XGBClassifier, XGBRegressor
            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        feature_names = X.columns.tolist()
        
        # Build preprocessor
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        
        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols))
        
        preprocessor = ColumnTransformer(transformers=transformers) if transformers else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == "classification" else None
        )
        
        # Define candidates
        if task_type == "regression":
            candidates = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=10),
                "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
            }
            if XGBOOST_AVAILABLE:
                candidates["XGBoost Regressor"] = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)
            scorer = r2_score
            score_type = "R² Score"
        else:
            candidates = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, max_depth=10),
                "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
                "SVM Classifier": SVC(kernel='rbf', random_state=42, probability=True),
                "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
                "Naive Bayes": GaussianNB()
            }
            if XGBOOST_AVAILABLE:
                candidates["XGBoost Classifier"] = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0, use_label_encoder=False, eval_metric='logloss')
            scorer = accuracy_score
            score_type = "Accuracy"
        
        # Run tournament
        results = []
        for name, estimator in candidates.items():
            try:
                model = Pipeline([("prep", preprocessor), ("model", estimator)]) if preprocessor else estimator
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = scorer(y_test, preds)
                results.append((name, float(score), model))
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue
        
        # Sort by score and get top 3
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Prefer scenario's designated model when we detected a known scenario
        scenario_model_type = None
        if scenario_id in SCENARIOS and SCENARIOS[scenario_id].get("task") == task_type:
            scenario_model_type = SCENARIOS[scenario_id].get("model_type")
        idx = _find_scenario_model_index(results, scenario_model_type) if scenario_model_type else None
        if idx is not None and idx > 0:
            # Move scenario model to front; keep order of rest
            chosen = results[idx]
            results = [chosen] + [r for i, r in enumerate(results) if i != idx]
        
        # Store the winning model and preprocessor (scenario model when preferred)
        if results:
            winning_model = results[0][2]
            winning_preprocessor = preprocessor
        
        top_models = []
        use_scenario_model = idx is not None
        for name, score, model in results[:3]:
            # Convert to percentage for classification
            display_score = round(score * 100, 1) if task_type == "classification" else round(score, 3)
            entry = {
                "name": name,
                "score": display_score,
                "score_type": score_type,
                "model_type": task_type,
                "explanation": get_model_explanation(name, task_type)
            }
            if use_scenario_model and name == results[0][0]:
                entry["scenario_recommended"] = True
                entry["scenario_name"] = scenario.get("name", "")
            top_models.append(entry)
    
    # Recommended model (top performer; scenario model when we preferred it)
    recommended_model = top_models[0] if top_models else None
    if recommended_model and scenario_id in SCENARIOS and task_type in ("clustering", "anomaly"):
        # Clustering/anomaly: single model; mark as scenario match when scenario fits
        st = SCENARIOS[scenario_id].get("model_type", "")
        if _scenario_model_matches(recommended_model["name"], st):
            recommended_model["scenario_recommended"] = True
            recommended_model["scenario_name"] = scenario.get("name", "")
    
    # Cache the winning model for demo predictions
    session_id = None
    if winning_model is not None:
        cache = get_model_cache()
        
        # Prepare feature information for the frontend
        feature_info = {
            "feature_names": feature_names,
            "num_features": len(feature_names)
        }
        
        # Store model in cache
        session_id = cache.store_model(
            model=winning_model,
            preprocessor=winning_preprocessor,
            scenario_id=scenario_id,
            scenario_data={
                "id": scenario_id,
                "name": scenario.get("name", "Unknown"),
                "description": scenario.get("description", ""),
                "icon": scenario.get("icon", "🤖"),
                "industry": scenario.get("industry", "General"),
            },
            feature_info=feature_info,
            task_type=task_type,
            model_name=recommended_model["name"] if recommended_model else "Unknown"
        )
    
    return {
        "scenario": {
            "id": scenario_id,
            "name": scenario.get("name", "Unknown"),
            "description": scenario.get("description", ""),
            "icon": scenario.get("icon", "🤖"),
            "industry": scenario.get("industry", "General"),
            "confidence": round(scenario_confidence * 100, 1)
        },
        "top_models": top_models,
        "recommended_model": recommended_model,
        "dataset_summary": dataset_summary,
        "task_type": task_type,
        "overall_confidence": round((scenario_confidence + (top_models[0]["score"] / 100 if task_type == "classification" else top_models[0]["score"])) / 2 * 100, 1) if top_models else 50.0,
        "session_id": session_id,  # NEW: For demo page predictions
        "feature_info": {  # NEW: For demo page input forms
            "feature_names": feature_names,
            "num_features": len(feature_names)
        }
    }


def get_all_scenarios() -> List[Dict[str, Any]]:
    """Return all available scenarios for showcase."""
    return [
        {
            "id": scenario_id,
            **scenario
        }
        for scenario_id, scenario in SCENARIOS.items()
    ]
