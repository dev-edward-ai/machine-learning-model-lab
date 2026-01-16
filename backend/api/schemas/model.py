from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Describes an available model endpoint."""

    name: str = Field(..., description="Internal identifier used in API requests")
    display_name: str = Field(..., description="Human readable model name")
    version: str = Field(default="1.0", description="Semantic version of the preset")
    task_type: str = Field(..., description="Primary ML task e.g. regression, classification")
    description: str = Field(..., description="Short summary of what the model does")


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class PredictionMetadata(BaseModel):
    rows: int
    columns: List[str]
    task_type: str
    target_column: Optional[str] = None


class PredictionResponse(BaseModel):
    model_name: str
    model_type: str
    predictions: List[Any]
    metadata: PredictionMetadata
    preview: Optional[List[Dict[str, Any]]] = None
    csv_base64: Optional[str] = Field(default=None, description="Base64 encoded CSV with predictions")
    csv_filename: Optional[str] = None
