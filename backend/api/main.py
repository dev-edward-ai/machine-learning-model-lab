from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers.models import router as models_router
from .routers.predict import router as predict_router

tags_metadata = [
    {
        "name": "models",
        "description": "Metadata about the built-in machine learning presets.",
    },
    {
        "name": "prediction",
        "description": "Endpoints for running datasets through selected models.",
    },
]

app = FastAPI(
    title="ML Model Serving API",
    version="0.1.0",
    description=(
        "Upload tabular datasets, choose a supported classical ML algorithm, "
        "and receive predictions plus optional previews or downloadable CSV outputs."
    ),
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "null",  # file:// origins become "null" in browsers
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router)
app.include_router(predict_router)


@app.get("/ping", tags=["health"])
def ping():
    return {"msg": "pong"}
