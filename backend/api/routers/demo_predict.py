"""
Demo Prediction Router

Handles live prediction requests from demo pages using cached models.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ..services.model_cache import get_model_cache


router = APIRouter(prefix="/demo", tags=["demo"])


class PredictionRequest(BaseModel):
    """Request model for live predictions."""
    features: Dict[str, Any] = Field(
        ...,
        description="Feature values as key-value pairs matching the trained model's input features",
        example={
            "rsi": 65.5,
            "macd": 0.05,
            "volume": 1000000,
            "moving_avg_50": 45000,
            "moving_avg_200": 42000
        }
    )


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: Any = Field(..., description="Model prediction (class label or numeric value)")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1) for classification tasks")
    probability: Optional[Dict[str, float]] = Field(None, description="Class probabilities for classification")
    explanation: Optional[str] = Field(None, description="Human-readable explanation of the prediction")
    task_type: str = Field(..., description="Type of ML task (classification, regression, etc.)")


@router.post("/predict/{session_id}", response_model=PredictionResponse)
async def predict_with_cached_model(
    session_id: str,
    request: PredictionRequest
) -> PredictionResponse:
    """
    Make a prediction using a cached trained model.
    
    Args:
        session_id: Unique session ID from previous analysis
        request: Feature values for prediction
    
    Returns:
        Prediction result with confidence/probability
    
    Raises:
        HTTPException: If session expired, not found, or prediction fails
    """
    cache = get_model_cache()
    
    # Retrieve cached model
    cached_data = cache.get_model(session_id)
    
    if cached_data is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Session not found or expired",
                "message": "Your analysis session has expired. Please re-run the analysis to get a new session.",
                "session_id": session_id
            }
        )
    
    try:
        # Extract model components
        model = cached_data['model']
        preprocessor = cached_data['preprocessor']
        task_type = cached_data['task_type']
        feature_info = cached_data['feature_info']
        scenario_data = cached_data['scenario_data']
        
        # Convert input features to DataFrame
        input_df = pd.DataFrame([request.features])
        
        # Validate features
        expected_features = feature_info.get('feature_names', [])
        if expected_features:
            missing_features = set(expected_features) - set(input_df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {', '.join(missing_features)}")
        
        # Make prediction
        if preprocessor:
            # Model has preprocessing pipeline
            prediction = model.predict(input_df)
        else:
            # Direct prediction (for models without preprocessing)
            prediction = model.predict(input_df)
        
        # Extract prediction value
        pred_value = prediction[0]
        
        # Get confidence/probability for classification
        confidence = None
        probability_dict = None
        explanation = None
        
        if task_type == "classification":
            # Try to get prediction probabilities
            try:
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(input_df)[0]
                    confidence = float(max(probas))
                    
                    # Get class labels
                    if hasattr(model, 'classes_'):
                        classes = model.classes_
                    else:
                        # For pipelines, get from the final estimator
                        final_estimator = model.named_steps.get('model', model)
                        classes = final_estimator.classes_ if hasattr(final_estimator, 'classes_') else None
                    
                    if classes is not None:
                        probability_dict = {
                            str(cls): float(prob)
                            for cls, prob in zip(classes, probas)
                        }
                        
                        # Create explanation
                        pred_label = str(pred_value)
                        explanation = f"Prediction: {pred_label} with {confidence*100:.1f}% confidence"
                    else:
                        confidence = float(max(probas))
                        explanation = f"Confidence: {confidence*100:.1f}%"
                else:
                    explanation = f"Prediction: {pred_value}"
            except Exception as e:
                print(f"Could not get probabilities: {e}")
                explanation = f"Prediction: {pred_value}"
        
        elif task_type == "regression":
            explanation = f"Predicted value: {float(pred_value):.2f}"
        
        else:
            # Clustering, anomaly detection, etc.
            explanation = f"Result: {pred_value}"
        
        # Convert numpy types to Python types
        if isinstance(pred_value, (np.integer, np.floating)):
            pred_value = float(pred_value)
        elif isinstance(pred_value, np.ndarray):
            pred_value = pred_value.tolist()
        
        return PredictionResponse(
            prediction=pred_value,
            confidence=confidence,
            probability=probability_dict,
            explanation=explanation,
            task_type=task_type
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid input",
                "message": str(e),
                "expected_features": feature_info.get('feature_names', [])
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "message": str(e)
            }
        )


@router.get("/session/{session_id}")
async def get_session_info(session_id: str) -> Dict[str, Any]:
    """
    Get information about a cached session.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        Session metadata (scenario, features, task type, etc.)
    """
    cache = get_model_cache()
    cached_data = cache.get_model(session_id)
    
    if cached_data is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired"
        )
    
    return {
        "session_id": session_id,
        "scenario_id": cached_data['scenario_id'],
        "scenario_data": cached_data['scenario_data'],
        "feature_info": cached_data['feature_info'],
        "task_type": cached_data['task_type'],
        "model_name": cached_data['model_name']
    }


@router.get("/cache/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics (for debugging/monitoring)."""
    cache = get_model_cache()
    return cache.get_cache_stats()
