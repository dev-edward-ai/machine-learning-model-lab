from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
import io

from .routers.models import router as models_router
from .routers.predict import router as predict_router
from .services.auto_model import recommend_and_run_best_model

tags_metadata = [
    {
        "name": "models",
        "description": "Metadata about the built-in machine learning presets.",
    },
    {
        "name": "prediction",
        "description": "Endpoints for running datasets through selected models.",
    },
    {
        "name": "automl",
        "description": "Automatic model detection and analysis.",
    },
]

app = FastAPI(
    title="AutoML Intelligence Platform",
    version="2.0.0",
    description=(
        "Professional AutoML platform with automatic model detection, real-world explanations, "
        "and support for 10+ machine learning algorithms including Linear/Logistic Regression, "
        "Decision Tree, KNN, SVM, Random Forest, K-Means, Naive Bayes, PCA, and XGBoost."
    ),
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Docker deployment
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(models_router)
app.include_router(predict_router)


@app.get("/ping", tags=["health"])
def ping():
    return {"msg": "pong", "status": "healthy"}


@app.post("/analyze", tags=["automl"])
async def analyze_dataset(
    file: UploadFile = File(..., description="CSV dataset to analyze"),
    target_col: Optional[str] = Form(None, description="Target column name (optional for clustering)"),
    user_intent: Optional[str] = Form("analyze data", description="User's goal description"),
    business_objective: Optional[str] = Form(None, description="Business objective category"),
):
    """
    Automatic model detection and analysis endpoint.
    Runs tournament across all 10+ available models and returns best fit with real-world explanations.
    """
    try:
        # Read CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        await file.close()
        
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
        
        # Add preview data (first 5 rows)
        if len(df) > 0:
            preview_df = df.head(5).copy()
            response["preview_data"] = preview_df.to_dict(orient="records")
        else:
            response["preview_data"] = []
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e)}
        )

