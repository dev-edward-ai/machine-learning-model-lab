"""
Business Insight Generator - FastAPI Backend
4 Hardcoded Scenario Detection & Model Training
Render-Ready Deployment
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import io
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from logic import detect_scenario, train_specific_model
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from logic import detect_scenario, train_specific_model

try:
    from api.services.scenario_trainers import DEDICATED_TRAINERS, LIVE_PREDICTORS
except ImportError:
    try:
        from services.scenario_trainers import DEDICATED_TRAINERS, LIVE_PREDICTORS
    except ImportError:
        DEDICATED_TRAINERS = {}
        LIVE_PREDICTORS = {}


# Initialize FastAPI
app = FastAPI(
    title="Business Insight Generator",
    description="4 Hardcoded Scenarios with Specialized ML Models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for current model (original 4 scenarios)
CURRENT_MODEL = None
CURRENT_SCENARIO = None

# Global storage for the 6 new dedicated-trainer scenarios
# key = scenario_key (e.g. 'HOUSE'), value = full trainer result dict
EXTRA_MODELS: dict = {}


# ============================================================================
# ROOT & HEALTH CHECK
# ============================================================================

@app.get("/ping", tags=["health"])
def ping():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Business Insight Generator is running"}


# ============================================================================
# STATIC FILE SERVING - samples only (frontend mounted at root at end of file)
# ============================================================================

samples_paths = ["/app/samples", "./samples", "../samples"]

for path in samples_paths:
    if os.path.exists(path):
        try:
            app.mount("/samples", StaticFiles(directory=path), name="samples")
            print(f"Mounted samples from: {path}")
            break
        except Exception as e:
            print(f"Failed to mount samples from {path}: {e}")


# ============================================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================================

@app.post("/analyze", tags=["analysis"])
async def analyze_file(file: UploadFile = File(...), target_col: str = None):
    """
    Upload a CSV file.
    Backend will:
    1. Detect the scenario based on column names
    2. Train the hardcoded model
    3. Return scenario_type + predictions
    """
    global CURRENT_MODEL, CURRENT_SCENARIO
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Detect scenario
        scenario = detect_scenario(df)
        
        # Train model
        result = train_specific_model(df, scenario)
        
        # Store for predictions
        CURRENT_MODEL = result
        CURRENT_SCENARIO = scenario
        
        # Prepare response
        response = {
            "scenario": scenario,
            "message": f"Successfully analyzed {scenario} dataset",
            "dataset_info": {
                "rows": len(df),
                "columns": list(df.columns),
                "feature_count": len(result['feature_cols'])
            },
            "model_info": {
                "features": result['feature_cols'],
                "accuracy": float(result.get('accuracy', result.get('r2_score', 0)))
            }
        }
        
        # Scenario-specific info
        if scenario == 'CRYPTO':
            response["model_name"] = "Logistic Regression"
            response["task_type"] = "Classification (Buy/Sell)"
            response["classes"] = result['classes']
            response["sample_prediction"] = {
                "prediction": "BUY (1)",
                "confidence": 0.85
            }
        
        elif scenario == 'MEDICAL':
            response["model_name"] = "K-Nearest Neighbors"
            response["task_type"] = "Classification (Risk Assessment)"
            response["classes"] = result['classes']
            response["sample_prediction"] = {
                "prediction": "Low Risk",
                "confidence": 0.78
            }
        
        elif scenario == 'CAR_PRICE':
            response["model_name"] = "Decision Tree Regressor"
            response["task_type"] = "Regression (Price Estimation)"
            response["price_range"] = {
                "min": float(result['y_min']),
                "max": float(result['y_max'])
            }
            response["sample_prediction"] = {
                "predicted_price": 25000,
                "confidence": 0.82
            }
        
        elif scenario == 'SALES':
            response["model_name"] = "Random Forest Regressor"
            response["task_type"] = "Regression (Revenue Prediction)"
            response["revenue_range"] = {
                "min": float(result['y_min']),
                "max": float(result['y_max'])
            }
            response["sample_prediction"] = {
                "predicted_revenue": 50000,
                "roi_multiplier": 2.5
            }
        
        return JSONResponse(response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post("/predict/crypto", tags=["predictions"])
async def predict_crypto(data: Dict[str, Any]):
    """
    Make a crypto trading prediction.
    Input: {"open": 65000, "close": 65500, "volume": 1000000, "rsi": 55}
    """
    if CURRENT_SCENARIO != 'CRYPTO' or CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="No CRYPTO model loaded. Upload a crypto dataset first.")
    
    try:
        model = CURRENT_MODEL['model']
        scaler = CURRENT_MODEL['scaler']
        features = CURRENT_MODEL['feature_cols']
        
        # Map frontend fields to training columns (case-insensitive)
        field_mapping = {
            'open': ['open', 'price', 'open_price'],
            'close': ['close', 'close_price', 'moving_average_7'],
            'volume': ['volume', 'vol'],
            'rsi': ['rsi', 'relative_strength'],
            'macd': ['macd', 'signal_strength']
        }
        
        values = []
        for feat in features:
            feat_lower = feat.lower()
            val = None
            # Direct match first
            for key, aliases in field_mapping.items():
                if feat_lower in aliases or feat_lower == key:
                    val = data.get(key, data.get(feat, 0))
                    break
            if val is None:
                val = data.get(feat, data.get(feat.lower(), 0))
            values.append(float(val) if val is not None else 0.0)
        
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        confidence = max(model.predict_proba(X_scaled)[0])
        
        signal = "STRONG BUY" if prediction == 1 and confidence > 0.7 else ("BUY" if prediction == 1 else "SELL")
        
        return {
            "signal": signal,
            "confidence": float(confidence),
            "class": CURRENT_MODEL['classes'][int(prediction)]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/medical", tags=["predictions"])
async def predict_medical(data: Dict[str, Any]):
    """
    Make a medical risk prediction.
    Input: {"age": 45, "bmi": 28.5, "bloodpressure": 120, "glucose": 125}
    """
    if CURRENT_SCENARIO != 'MEDICAL' or CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="No MEDICAL model loaded. Upload a medical dataset first.")
    
    try:
        model = CURRENT_MODEL['model']
        scaler = CURRENT_MODEL['scaler']
        features = CURRENT_MODEL['feature_cols']
        
        # Map frontend fields to training columns (case-insensitive)
        field_mapping = {
            'age': ['age'],
            'bmi': ['bmi', 'body_mass_index'],
            'bloodpressure': ['bloodpressure', 'resting_bp', 'bp', 'blood_pressure'],
            'glucose': ['glucose', 'fasting_blood_sugar', 'blood_sugar', 'cholesterol'],
            'cholesterol': ['cholesterol', 'chol']
        }
        
        values = []
        for feat in features:
            feat_lower = feat.lower()
            val = None
            # Find matching frontend key
            for key, aliases in field_mapping.items():
                if feat_lower in aliases or feat_lower == key:
                    val = data.get(key, data.get(feat, 0))
                    break
            if val is None:
                val = data.get(feat, data.get(feat.lower(), 0))
            values.append(float(val) if val is not None else 0.0)
        
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]
        confidence = max(model.predict_proba(X_scaled)[0])
        
        return {
            "risk_level": CURRENT_MODEL['classes'][int(prediction)],
            "confidence": float(confidence)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/car_price", tags=["predictions"])
async def predict_car_price(data: Dict[str, Any]):
    """
    Make a car price prediction.
    Input: {"year": 2015, "mileage": 80000, "horsepower": 200}
    """
    if CURRENT_SCENARIO != 'CAR_PRICE' or CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="No CAR_PRICE model loaded. Upload a car dataset first.")
    
    try:
        model = CURRENT_MODEL['model']
        scaler = CURRENT_MODEL['scaler']
        features = CURRENT_MODEL['feature_cols']
        
        # Map frontend fields to training columns (case-insensitive)
        field_mapping = {
            'year': ['year', 'model_year', 'yr'],
            'mileage': ['mileage', 'miles', 'odometer', 'km'],
            'horsepower': ['horsepower', 'hp', 'engine_size', 'engine', 'power']
        }
        
        values = []
        for feat in features:
            feat_lower = feat.lower()
            val = None
            # Find matching frontend key
            for key, aliases in field_mapping.items():
                if feat_lower in aliases or feat_lower == key:
                    val = data.get(key, data.get(feat, 0))
                    break
            if val is None:
                val = data.get(feat, data.get(feat.lower(), 0))
            values.append(float(val) if val is not None else 0.0)
        
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]
        
        return {
            "estimated_price": float(prediction),
            "currency": "USD",
            "confidence": float(CURRENT_MODEL.get('r2_score', 0.8))
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/sales", tags=["predictions"])
async def predict_sales(data: Dict[str, Any]):
    """
    Make a sales/revenue prediction.
    Input: {"adspend": 5000, "socialclicks": 10000, "season": "summer"}
    """
    if CURRENT_SCENARIO != 'SALES' or CURRENT_MODEL is None:
        raise HTTPException(status_code=400, detail="No SALES model loaded. Upload a sales dataset first.")
    
    try:
        model = CURRENT_MODEL['model']
        scaler = CURRENT_MODEL['scaler']
        features = CURRENT_MODEL['feature_cols']
        
        # Map frontend fields to training columns (case-insensitive)
        field_mapping = {
            'adspend': ['adspend', 'ad_spend', 'budget', 'spend', 'marketing_budget'],
            'socialclicks': ['socialclicks', 'social_clicks', 'clicks', 'impressions', 'audience_size']
        }
        
        values = []
        for feat in features:
            feat_lower = feat.lower()
            val = None
            # Find matching frontend key
            for key, aliases in field_mapping.items():
                if feat_lower in aliases or feat_lower == key:
                    val = data.get(key, data.get(feat, 0))
                    break
            if val is None:
                val = data.get(feat, data.get(feat.lower(), 0))
            values.append(float(val) if val is not None else 0.0)
        
        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]
        budget = data.get('adspend', 5000)
        roi = prediction / budget if budget > 0 else 0
        
        return {
            "predicted_revenue": float(prediction),
            "budget": float(budget),
            "roi_multiplier": float(roi),
            "expected_profit": float(prediction - budget)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# ============================================================================
# NEW SCENARIO ANALYSIS ENDPOINT (6 dedicated trainers)
# ============================================================================

# Maps frontend scenario key → DEDICATED_TRAINERS key
_NEW_SCENARIO_MAP = {
    'HOUSE':    'regression_housing',
    'BANKNOTE': 'banknote_authentication',
    'CHURN':    'customer_churn',
    'SPAM':     'sms_spam',
    'SEGMENT':  'customer_segments',
    'STOCK':    'stock_sectors',
}

@app.post("/analyze/new", tags=["analysis"])
async def analyze_new_scenario(
    file: UploadFile = File(...),
    scenario_key: str = Form(None)
):
    """
    Train one of the 6 new dedicated ML models.
    scenario_key must be one of: HOUSE, BANKNOTE, CHURN, SPAM, SEGMENT, STOCK
    """
    global EXTRA_MODELS

    trainer_key = _NEW_SCENARIO_MAP.get((scenario_key or '').upper())
    if trainer_key is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario_key '{scenario_key}'. "
                   f"Valid options: {list(_NEW_SCENARIO_MAP.keys())}"
        )

    trainer_fn = DEDICATED_TRAINERS.get(trainer_key)
    if trainer_fn is None:
        raise HTTPException(
            status_code=500,
            detail=f"Dedicated trainer '{trainer_key}' not found. Check installation."
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if df.empty:
            raise ValueError("CSV file is empty")

        result = trainer_fn(df)
        EXTRA_MODELS[scenario_key.upper()] = result

        return JSONResponse({
            "scenario": scenario_key.upper(),
            "model_name": result["model_name"],
            "task_type": result["task_type"],
            "metrics": result["metrics"],
            "feature_cols": result.get("feature_cols", []),
            "explained": result.get("explained", ""),
            "dataset_info": {
                "rows": len(df),
                "columns": list(df.columns)
            },
            # Pass through scenario-specific extras (centroids, scatter_data, etc.)
            "extras": {
                k: v for k, v in result.items()
                if k not in ("model", "scaler", "pca", "preprocessor", "explained")
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training error: {str(e)}")


# ============================================================================
# NEW PREDICTION ENDPOINTS
# ============================================================================

@app.post("/predict/house", tags=["predictions"])
async def predict_house(data: Dict[str, Any]):
    """
    Predict house price.
    Input: {"square_footage": 1800, "bedrooms": 3, "bathrooms": 2,
             "location_score": 7.5, "house_age": 15}
    """
    m = EXTRA_MODELS.get('HOUSE')
    if m is None:
        raise HTTPException(status_code=400, detail="No HOUSE model loaded. Upload regression_housing.csv first.")
    try:
        feature_cols = m['feature_cols']
        scaler = m['scaler']
        model = m['model']
        X = np.array([[float(data.get(f, data.get(f.replace('_', ''), 0) or 0))
                       for f in feature_cols]])
        X_s = scaler.transform(X)
        price = float(model.predict(X_s)[0])
        return {"predicted_price": round(max(price, 0), 2), "currency": "USD",
                "r2_score": m['metrics'].get('r2_score', 0)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/banknote", tags=["predictions"])
async def predict_banknote(data: Dict[str, Any]):
    """
    Classify banknote as real or fake.
    Input: {"variance": 2.0, "skewness": 3.5, "curtosis": -1.2, "entropy": 0.5}
    """
    m = EXTRA_MODELS.get('BANKNOTE')
    if m is None:
        raise HTTPException(status_code=400, detail="No BANKNOTE model loaded. Upload banknote_authentication.csv first.")
    try:
        feature_cols = m['feature_cols']
        scaler = m['scaler']
        model = m['model']
        X = np.array([[float(data.get(f, data.get(f.lower(), 0) or 0))
                       for f in feature_cols]])
        X_s = scaler.transform(X)
        pred_idx = int(model.predict(X_s)[0])
        proba = model.predict_proba(X_s)[0]
        classes = m.get('label_classes', ['fake', 'real'])
        label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
        return {
            "predicted_class": label,
            "authentic": bool(pred_idx == len(classes) - 1),
            "confidence": round(float(np.max(proba)) * 100, 1),
            "probabilities": {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/churn", tags=["predictions"])
async def predict_churn(data: Dict[str, Any]):
    """
    Predict customer churn probability.
    Input: {"tenure": 12, "monthly_charges": 65.0, "contract_type": 0,
             "support_tickets": 2, "payment_method": 1}
    """
    m = EXTRA_MODELS.get('CHURN')
    if m is None:
        raise HTTPException(status_code=400, detail="No CHURN model loaded. Upload customer_churn.csv first.")
    try:
        feature_cols = m['feature_cols']
        model = m['model']
        X = np.array([[float(data.get(f, data.get(f.lower().replace('_', ''), 0) or 0))
                       for f in feature_cols]])
        proba = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])
        return {
            "churn_probability": round(proba * 100, 1),
            "predicted": "CHURN" if pred == 1 else "RETAIN",
            "risk_level": "High" if proba >= 0.7 else ("Medium" if proba >= 0.4 else "Low")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/spam", tags=["predictions"])
async def predict_spam(data: Dict[str, Any]):
    """
    Classify SMS as spam or ham.
    Input: {"text": "Congratulations! You won a free prize!"}
    Works with both raw-text models (TF-IDF pipeline) and
    engineered-feature models (Logistic Regression).
    """
    m = EXTRA_MODELS.get('SPAM')
    if m is None:
        raise HTTPException(status_code=400, detail="No SPAM model loaded. Upload sms_spam.csv first.")
    try:
        text = str(data.get('text', ''))
        if not text.strip():
            raise ValueError("text field is required")

        model_type = m.get('model_type', 'tfidf_pipeline')
        label_map  = m.get('label_map', {0: 'ham', 1: 'spam'})

        if model_type == 'tfidf_pipeline':
            pipeline = m['model']
            proba = pipeline.predict_proba([text])[0]
            pred  = int(pipeline.predict([text])[0])
        else:
            # numeric LR model: derive heuristic features from text
            import re
            words = text.split()
            row_vals = {
                'word_count':        len(words),
                'special_chars':     len(re.findall(r'[^\w\s]', text)),
                'capital_ratio':     sum(1 for c in text if c.isupper()) / max(len(text), 1),
                'has_url':           int(bool(re.search(r'http|www\.', text, re.I))),
                'has_money_words':   int(bool(re.search(r'free|win|prize|cash|money|reward', text, re.I))),
                'has_urgency_words': int(bool(re.search(r'urgent|immediately|now|call|act', text, re.I))),
                'exclamation_count': text.count('!'),
                'number_count':      len(re.findall(r'\d+', text)),
            }
            feature_cols = m['feature_cols']
            scaler       = m['scaler']
            x = np.array([[float(row_vals.get(f, 0.0)) for f in feature_cols]])
            x_scaled = scaler.transform(x)
            model = m['model']
            proba = model.predict_proba(x_scaled)[0]
            pred  = int(model.predict(x_scaled)[0])

        # Map integer key → string (label_map stored as {0: 'ham', 1: 'spam'})
        label = label_map.get(pred) or label_map.get(str(pred)) or ('spam' if pred == 1 else 'ham')
        return {
            "predicted":        label,
            "is_spam":          bool(pred == 1),
            "spam_probability": round(float(proba[1]) * 100, 1),
            "top_spam_words":   m.get('top_spam_words', [])[:8],
            "model_type":       model_type,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/segment", tags=["predictions"])
async def predict_segment(data: Dict[str, Any]):
    """
    Assign a customer to a cluster.
    Input: {"annual_income": 60000, "spending_score": 72, "purchase_frequency": 18}
    """
    m = EXTRA_MODELS.get('SEGMENT')
    if m is None:
        raise HTTPException(status_code=400, detail="No SEGMENT model loaded. Upload clustering_customers.csv first.")
    try:
        feature_cols = m['feature_cols']
        scaler = m['scaler']
        model = m['model']
        X = np.array([[float(data.get(f, data.get(f.lower(), 0) or 0))
                       for f in feature_cols]])
        X_s = scaler.transform(X)
        cluster = int(model.predict(X_s)[0])
        dist = float(np.linalg.norm(model.cluster_centers_[cluster] - X_s))
        confidence = round(1.0 / (1.0 + dist), 3)
        return {"cluster": cluster, "assignment_confidence": confidence,
                "optimal_k": m['metrics'].get('optimal_k'),
                "silhouette_score": m['metrics'].get('silhouette_score')}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/stock", tags=["predictions"])
async def predict_stock(data: Dict[str, Any]):
    """
    Project a new stock's features through the fitted PCA and return its 2D position.
    Input: keys match whatever numeric feature columns were in the training CSV.
    Example: {"tech_score": 8, "value_score": 4, "pe_ratio": 35, "rsi": 65, ...}
    """
    m = EXTRA_MODELS.get('STOCK')
    if m is None:
        raise HTTPException(status_code=400, detail="No STOCK model loaded. Upload stock_sectors.csv first.")
    try:
        pca = m['model']
        scaler = m['scaler']
        feature_cols = m['feature_cols']
        x = np.array([[float(data.get(f, 0.0)) for f in feature_cols]])
        x_scaled = scaler.transform(x)
        coords = pca.transform(x_scaled)[0]
        evr = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]
        return {
            "pc1": round(float(coords[0]), 4),
            "pc2": round(float(coords[1]), 4) if len(coords) > 1 else 0.0,
            "variance_explained": evr,
            "feature_cols": feature_cols,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# INFO ENDPOINT
# ============================================================================

@app.get("/current_scenario", tags=["info"])
def get_current_scenario():
    """Get the currently loaded scenario."""
    if CURRENT_SCENARIO is None:
        return {"scenario": None, "message": "No model loaded yet. Upload a CSV file first."}
    
    return {
        "scenario": CURRENT_SCENARIO,
        "model_loaded": True
    }


# ============================================================================
# FRONTEND - mounted at root LAST so all API routes take priority
# ============================================================================

frontend_paths = ["/app/frontend", "./frontend", "../frontend"]
for path in frontend_paths:
    if os.path.exists(path):
        try:
            app.mount("/", StaticFiles(directory=path, html=True), name="frontend")
            print(f"Mounted frontend at / from: {path}")
            break
        except Exception as e:
            print(f"Failed to mount frontend from {path}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
