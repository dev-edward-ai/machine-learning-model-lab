import io
import base64
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..schemas.model import BusinessInsights, PredictionMetadata, PredictionResponse
from ..services.prediction import model_registry
from ..services.auto_model import recommend_and_run_best_model

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
async def run_prediction(
    objective: Optional[str] = Form(
        None, description="High-level business goal (e.g., churn, revenue, fraud)"
    ),
    model_name: str = Form(..., description="Name reported by /models"),
    use_preview: bool = Form(False, description="Include first 5 rows with predictions"),
    return_csv: bool = Form(False, description="Attach base64 CSV with predictions"),
    target_column: Optional[str] = Form(
        None,
        description="Optional explicit target/label column for supervised models",
    ),
    file: UploadFile = File(..., description="CSV dataset to score"),
) -> PredictionResponse:
    dataset = await _read_csv(file)
    result = model_registry.predict(
        model_name=model_name,
        dataset=dataset,
        use_preview=use_preview,
        return_csv=return_csv,
        target_column=target_column,
    )
    metadata = PredictionMetadata(**result.metadata)
    business_insights = _build_business_insights(
        objective=objective,
        model_type=result.model_type,
        predictions=result.predictions,
    )

    return PredictionResponse(
        model_name=result.model_name,
        model_type=result.model_type,
        predictions=result.predictions,
        metadata=metadata,
        preview=result.preview,
        csv_base64=result.csv_base64,
        csv_filename=result.csv_filename,
        business_insights=business_insights,
    )


@router.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(..., description="CSV dataset to analyze"),
    target_col: Optional[str] = Form(None, description="Target column name (optional for clustering)"),
    user_intent: Optional[str] = Form("analyze data", description="User's goal description"),
    business_objective: Optional[str] = Form(None, description="Business objective category"),
):
    """
    Automatic model detection and analysis endpoint.
    Runs tournament across all available models and returns best fit with explanations.
    """
    try:
        # Read CSV
        df = await _read_csv(file)
        
        # Run AutoML
        result = recommend_and_run_best_model(
            df=df,
            target_col=target_col,
            user_intent=user_intent or "analyze data",
            business_objective=business_objective
        )
        
        # Prepare response
        response = {
            "recommended_model": result["recommended_model_name"],
            "recommended_model_name": result["recommended_model_name"],
            "task_type": result["task_type"],
            "metric_value": result["metric_value"],
            "score": result["metric_value"],
            "reasoning": result["reasoning"],
            "business_insights": result.get("business_summary", {}),
            "model_explanation": result.get("model_explanation", {}),
        }
        
        # Add preview data (first 5 rows with predictions)
        if len(df) > 0:
            preview_df = df.head(5).copy()
            response["preview_data"] = preview_df.to_dict(orient="records")
        else:
            response["preview_data"] = []
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)}
        )


async def _read_csv(upload_file: UploadFile) -> pd.DataFrame:
    try:
        content = await upload_file.read()
        if not content:
            raise ValueError("File is empty")
        buffer = io.BytesIO(content)
        dataframe = pd.read_csv(buffer)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read CSV: {exc}",
        ) from exc
    finally:
        await upload_file.close()
    return dataframe


def _normalize_objective(objective: Optional[str]) -> Optional[str]:
    if not objective:
        return None
    text = objective.strip().lower()
    if "churn" in text:
        return "churn"
    if "revenue" in text or "sales" in text:
        return "sales"
    if "fraud" in text or "anomaly" in text or "outlier" in text:
        return "fraud"
    return None


def _build_business_insights(
    *, objective: Optional[str], model_type: str, predictions: List[Any]
) -> Optional[BusinessInsights]:
    intent = _normalize_objective(objective)
    if not intent:
        return None

    headline = "Prediction complete."
    detailed_insight = None
    recommended_action = None

    if intent == "churn":
        at_risk = sum(1 for p in predictions if str(p) == "1" or p == 1)
        total = len(predictions)
        pct = (at_risk / total * 100) if total else 0
        headline = f"⚠️ Alert: {at_risk} customers are at risk of churning."
        detailed_insight = f"{pct:.1f}% of analyzed customers are flagged as at risk."
        recommended_action = "Recommended Action: Launch retention campaign for these specific users."
    elif intent == "sales":
        try:
            total_revenue = sum(float(p) for p in predictions)
        except Exception:
            total_revenue = 0.0
        headline = f"💰 Forecast: Total projected revenue is ${total_revenue:,.2f}."
        detailed_insight = "Aggregated forecast across submitted records."
        recommended_action = "Recommended Action: Allocate budget to capture this demand."
    elif intent == "fraud":
        anomalies = sum(1 for p in predictions if p == -1 or str(p) == "-1")
        headline = f"🛡️ Security Alert: {anomalies} suspicious transactions detected."
        detailed_insight = "Anomalies flagged for further investigation."

    return BusinessInsights(
        headline=headline,
        detailed_insight=detailed_insight,
        recommended_action=recommended_action,
        predictions=predictions,
    )
