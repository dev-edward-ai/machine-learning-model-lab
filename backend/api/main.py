"""
Business Insight Generator - FastAPI Backend
4 Hardcoded Scenario Detection & Model Training
Render-Ready Deployment
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
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
    # Fallback import from parent
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from logic import detect_scenario, train_specific_model


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

# Global storage for current model
CURRENT_MODEL = None
CURRENT_SCENARIO = None

# Mount static files for production (Render deployment)
# Check multiple possible paths for frontend and samples
frontend_paths = ["./frontend", "../frontend", "/app/frontend"]
samples_paths = ["./samples", "../samples", "/app/samples"]

for path in frontend_paths:
    if os.path.exists(path):
        app.mount("/frontend", StaticFiles(directory=path), name="frontend")
        break

for path in samples_paths:
    if os.path.exists(path):
        app.mount("/samples", StaticFiles(directory=path), name="samples")
        break


# ============================================================================
# ROOT & HEALTH CHECK
# ============================================================================

@app.get("/", tags=["root"])
def root():
    """Redirect to frontend."""
    return FileResponse("frontend/index.html") if os.path.exists("frontend/index.html") else {"message": "InsightAI API", "docs": "/docs"}


@app.get("/ping", tags=["health"])
def ping():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Business Insight Generator is running"}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
