"""
Smart Dispatcher System v2.0

Enhanced with ScenarioFingerprinting Engine to enforce domain logic over raw metrics.
Prevents lazy KNN selection through intelligent scenario detection and heuristic scoring.
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
from .scenario_fingerprinting import ScenarioFingerprinting, HeuristicScorer
from .scenario_trainers import DEDICATED_TRAINERS  # scenario-specific trainer registry


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
    },

    # ── 6 new production scenarios (each has a dedicated trainer) ─────────
    "regression_housing": {
        "name": "House Price Estimator",
        "description": "Linear price prediction from structural and location features",
        "model_type": "Linear Regression",
        "task": "regression",
        "icon": "🏡",
        "keywords": ["square_footage", "bedrooms", "bathrooms", "location_score", "house_age", "sqft", "price"],
        "industry": "Real Estate",
        "dedicated_trainer": "regression_housing"
    },
    "customer_segments": {
        "name": "Customer Segmentation",
        "description": "Auto-k K-Means segmentation with silhouette optimisation",
        "model_type": "KMeans",
        "task": "clustering",
        "icon": "👥",
        "keywords": ["annual_income", "spending_score", "purchase_frequency", "customer", "segment", "clv"],
        "industry": "Retail/E-commerce",
        "dedicated_trainer": "customer_segments"
    },
    # Existing scenarios enhanced with dedicated trainers
    "banknote_authentication": {
        "name": "Fake Banknote Detector",
        "description": "Precision boundary classification for counterfeit detection",
        "model_type": "SVM Classifier",
        "task": "classification",
        "icon": "💵",
        "keywords": ["variance", "skewness", "curtosis", "entropy", "authentic"],
        "industry": "Security/Banking",
        "dedicated_trainer": "banknote_authentication"
    },
    "customer_churn_v2": {
        "name": "Customer Churn Predictor",
        "description": "XGBoost churn prediction with class-imbalance handling",
        "model_type": "XGBoost Classifier",
        "task": "classification",
        "icon": "📊",
        "keywords": ["tenure", "monthly_charges", "contract_type", "support_tickets", "payment_method"],
        "industry": "SaaS/Subscription",
        "dedicated_trainer": "customer_churn"
    },
    "sms_spam_v2": {
        "name": "SMS Spam Detector",
        "description": "TF-IDF pipeline + Multinomial Naive Bayes text classifier",
        "model_type": "Multinomial Naive Bayes",
        "task": "text_classification",
        "icon": "📱",
        "keywords": ["message", "text", "sms", "spam", "ham", "label", "content"],
        "industry": "Communications",
        "dedicated_trainer": "sms_spam"
    },
    "stock_sectors_v2": {
        "name": "Stock Market Visualizer",
        "description": "PCA 2-component projection for sector cluster visualisation",
        "model_type": "PCA",
        "task": "dimensionality_reduction",
        "icon": "📉",
        "keywords": ["stock", "sector", "tech_score", "market", "beta", "volatility", "ticker"],
        "industry": "Finance/Investment",
        "dedicated_trainer": "stock_sectors"
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





def generate_ui_config(scenario_info: Dict, recommended_model: Dict, 
                      df: pd.DataFrame, target_col: Optional[str]) -> Dict[str, Any]:
    """Generate configuration for interactive UI components based on scenario."""
    
    ui_component = scenario_info.get("ui_component", "GeneralResults")
    
    base_config = {
        "component_type": ui_component,
        "title": f"Interactive {scenario_info.get('scenario_name', 'Analysis')}",
        "features": list(df.columns),
        "target": target_col,
        "sample_data": df.head(3).to_dict('records')
    }
    
    # Component-specific configurations
    if ui_component == "LoanOfficerDashboard":
        return {
            **base_config,
            "input_fields": [
                {"name": "income", "type": "number", "label": "Annual Income ($)", "min": 0, "max": 200000},
                {"name": "credit_score", "type": "number", "label": "Credit Score", "min": 300, "max": 850},
                {"name": "debt_to_income", "type": "number", "label": "Debt-to-Income Ratio", "min": 0, "max": 1, "step": 0.01},
                {"name": "employment_years", "type": "number", "label": "Years Employed", "min": 0, "max": 50}
            ],
            "prediction_endpoint": "/predict/decision-tree-live",
            "result_format": "approval_status"
        }
        
    elif ui_component == "TradingTerminal":
        return {
            **base_config,
            "chart_config": {
                "type": "candlestick",
                "indicators": ["rsi", "macd", "moving_average"],
                "timeframe": "1H"
            },
            "signal_display": {
                "buy_color": "#10B981",
                "sell_color": "#EF4444", 
                "neutral_color": "#6B7280"
            },
            "live_updates": True
        }
        
    elif ui_component == "ClusteringVisualization":
        return {
            **base_config,
            "plot_config": {
                "type": "scatter2d",
                "x_axis": df.columns[0],
                "y_axis": df.columns[1] if len(df.columns) > 1 else df.columns[0],
                "color_by": "cluster",
                "interactive": True
            },
            "cluster_info": {
                "n_clusters": len(set(recommended_model.get("cluster_assignments", [0]))),
                "cluster_centers": "calculated"
            }
        }
        
    elif ui_component == "PricingCalculator":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        input_fields = []
        for col in numeric_cols[:5]:  # Limit to 5 most important features
            if col != target_col:
                input_fields.append({
                    "name": col,
                    "type": "number", 
                    "label": col.replace('_', ' ').title(),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                })
        
        return {
            **base_config,
            "input_fields": input_fields,
            "prediction_format": "currency",
            "confidence_interval": True
        }
    
    # Default general configuration
    return {
        **base_config,
        "component_type": "GeneralResults",
        "show_metrics": True,
        "show_feature_importance": True
    }


def analyze_dataset_characteristics(df: pd.DataFrame, target_col: Optional[str]) -> Dict[str, Any]:
    """Analyze dataset for the 'Data DNA' left panel."""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    characteristics = {
        "dimensions": {
            "rows": len(df),
            "columns": len(df.columns),
            "features": len(df.columns) - (1 if target_col else 0)
        },
        
        "data_types": {
            "numeric": len(numeric_cols),
            "categorical": len(categorical_cols),
            "missing_values": df.isnull().sum().sum()
        },
        
        "data_quality": {
            "completeness": round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1),
            "duplicates": df.duplicated().sum(),
            "unique_ratio": round(df.nunique().mean() / len(df) * 100, 1)
        },
        
        "target_analysis": {}
    }
    
    if target_col and target_col in df.columns:
        target_series = df[target_col]
        characteristics["target_analysis"] = {
            "type": "binary" if target_series.nunique() == 2 else "continuous" if target_series.dtype in ['float64', 'int64'] else "categorical",
            "unique_values": target_series.nunique(),
            "distribution": target_series.value_counts().head().to_dict()
        }
    
    return characteristics


def generate_model_insights(recommended_model: Dict, scenario_info: Dict) -> Dict[str, Any]:
    """Generate insights for the right panel."""
    
    return {
        "model_choice": {
            "name": recommended_model["name"],
            "confidence": recommended_model["score"],
            "reasoning": f"Selected for {scenario_info.get('scenario_name', 'analysis')} based on domain expertise and performance."
        },
        
        "business_impact": {
            "accuracy_meaning": f"{recommended_model['score']:.1f}% accuracy means reliable predictions for business decisions.",
            "use_cases": [
                "Real-time decision support",
                "Automated processing",
                "Risk assessment",
                "Performance optimization"
            ]
        },
        
        "technical_details": {
            "algorithm_family": recommended_model.get("model_type", "supervised"),
            "interpretability": "High" if "Tree" in recommended_model["name"] else "Medium",
            "scalability": "Good",
            "training_time": "Fast"
        }
    }


def generate_session_id() -> str:
    """Generate unique session ID for tracking."""
    import uuid
    return str(uuid.uuid4())[:8]


def smart_dispatch(df, target_col=None, business_objective=None):
    """
    Enhanced smart dispatch with scenario detection and confidence scoring.
    
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

    # ── Dedicated-trainer fast path ───────────────────────────────────────
    # When a known scenario is detected with sufficient confidence AND a
    # specialised trainer exists, bypass the generic tournament entirely.
    # This enforces domain-correct model selection (e.g. TF-IDF+NB for SMS,
    # PCA for stocks) rather than letting the tournament pick generically.
    dedicated_trainer_key = scenario.get("dedicated_trainer")
    if dedicated_trainer_key and dedicated_trainer_key in DEDICATED_TRAINERS and scenario_confidence >= 0.3:
        try:
            trainer_fn = DEDICATED_TRAINERS[dedicated_trainer_key]
            trainer_result = trainer_fn(df)

            t_task = trainer_result["task_type"]
            t_model_name = trainer_result["model_name"]
            t_metrics = trainer_result["metrics"]
            primary_score = t_metrics.get("primary_score", 0.0)
            score_label = t_metrics.get("score_label", "Score")

            # Reuse model_cache for live-prediction endpoints
            cache = get_model_cache()
            feature_names = trainer_result.get("feature_cols", [])
            session_id = cache.store_model(
                model=trainer_result["model"],
                preprocessor=trainer_result.get("scaler"),
                scenario_id=scenario_id,
                scenario_data={
                    "id": scenario_id,
                    "name": scenario.get("name", "Unknown"),
                    "description": scenario.get("description", ""),
                    "icon": scenario.get("icon", "🤖"),
                    "industry": scenario.get("industry", "General"),
                },
                feature_info={"feature_names": feature_names, "num_features": len(feature_names)},
                task_type=t_task,
                model_name=t_model_name,
            )

            recommended_model = {
                "name": t_model_name,
                "score": primary_score,
                "score_type": score_label,
                "model_type": t_task,
                "explanation": get_model_explanation(t_model_name, t_task),
                "scenario_recommended": True,
                "scenario_name": scenario.get("name", ""),
                "dedicated_trainer": True,
            }

            dataset_summary = {
                "num_rows": len(df),
                "num_cols": len(df.columns),
                "num_numeric": len(df.select_dtypes(include=[np.number]).columns),
                "num_categorical": len(df.select_dtypes(exclude=[np.number]).columns),
                "missing_values": int(df.isnull().sum().sum()),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            }

            return {
                "scenario": {
                    "id": scenario_id,
                    "name": scenario.get("name", "Unknown"),
                    "description": scenario.get("description", ""),
                    "icon": scenario.get("icon", "🤖"),
                    "industry": scenario.get("industry", "General"),
                    "confidence": round(scenario_confidence * 100, 1),
                },
                "top_models": [recommended_model],
                "recommended_model": recommended_model,
                "dataset_summary": dataset_summary,
                "task_type": t_task,
                "overall_confidence": round(scenario_confidence * 100, 1),
                "session_id": session_id,
                "feature_info": {"feature_names": feature_names, "num_features": len(feature_names)},
                # Extra per-trainer payloads passed through transparently
                "trainer_extras": {
                    k: v
                    for k, v in trainer_result.items()
                    if k not in ("model", "scaler", "pca", "preprocessor")
                },
            }
        except Exception as dedicated_err:
            # Degrade gracefully to generic tournament if trainer fails
            print(f"[dispatcher] Dedicated trainer '{dedicated_trainer_key}' failed: {dedicated_err}. "
                  "Falling back to generic tournament.")

    # ── Generic tournament path (unchanged) ───────────────────────────────
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

    elif task_type == "text_classification":
        # Fallback: route through the SMS/text dedicated trainer (handles both
        # raw-text and feature-engineered CSV formats).
        from .scenario_trainers import train_sms_spam_detector
        result = train_sms_spam_detector(df)
        winning_model = result["model"]
        feature_names = [result.get("text_col") or "text"]
        top_models = [{
            "name": result["model_name"],
            "score": result["metrics"]["primary_score"],
            "score_type": result["metrics"]["score_label"],
            "model_type": "text_classification",
            "explanation": get_model_explanation(result["model_name"], task_type),
            "dedicated_trainer": True,
        }]

    elif task_type == "dimensionality_reduction":
        # Fallback: run PCA reduction via the stock visualizer trainer.
        from .scenario_trainers import train_stock_pca_visualizer
        result = train_stock_pca_visualizer(df)
        winning_model = result["model"]
        feature_names = result["feature_cols"]
        top_models = [{
            "name": result["model_name"],
            "score": result["metrics"]["primary_score"],
            "score_type": result["metrics"]["score_label"],
            "model_type": "dimensionality_reduction",
            "explanation": get_model_explanation(result["model_name"], task_type),
            "dedicated_trainer": True,
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
        if idx is not None and int(idx) > 0:
            # Move scenario model to front; keep order of rest
            idx = int(idx)
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
