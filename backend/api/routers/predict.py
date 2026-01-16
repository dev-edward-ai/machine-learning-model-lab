import io
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..schemas.model import PredictionMetadata, PredictionResponse
from ..services.prediction import model_registry

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictionResponse)
async def run_prediction(
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
    return PredictionResponse(
        model_name=result.model_name,
        model_type=result.model_type,
        predictions=result.predictions,
        metadata=metadata,
        preview=result.preview,
        csv_base64=result.csv_base64,
        csv_filename=result.csv_filename,
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
