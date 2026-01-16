from fastapi import APIRouter

from ..schemas.model import ModelInfo, ModelListResponse
from ..services.prediction import model_registry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    models = [
        ModelInfo(
            name=config.name,
            display_name=config.display_name,
            version=config.version,
            task_type=config.task_type,
            description=config.description,
        )
        for config in model_registry.list_models()
    ]
    return ModelListResponse(models=models)