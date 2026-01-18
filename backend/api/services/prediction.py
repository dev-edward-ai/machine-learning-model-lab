from __future__ import annotations

import base64
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:  # Optional dependency
    xgboost_module = importlib.import_module("xgboost")
    XGBClassifier = getattr(xgboost_module, "XGBClassifier", None)  # type: ignore[attr-defined]
    XGBRegressor = getattr(xgboost_module, "XGBRegressor", None)  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore


TaskType = Literal[
    "regression",
    "classification",
    "clustering",
    "dimensionality_reduction",
    "anomaly",
]


@dataclass(slots=True)
class ModelConfig:
    name: str
    display_name: str
    version: str
    task_type: TaskType
    description: str
    estimator_factory: Callable[[], Any]

    @property
    def requires_target(self) -> bool:
        return self.task_type in {"regression", "classification"}


@dataclass(slots=True)
class PredictionResult:
    model_name: str
    model_type: TaskType
    predictions: List[Any]
    metadata: Dict[str, Any]
    preview: Optional[List[Dict[str, Any]]]
    csv_base64: Optional[str]
    csv_filename: Optional[str]


class ModelRegistry:
    """Keeps track of supported estimators and executes predictions."""

    def __init__(self) -> None:
        self._registry = self._build_registry()

    def list_models(self) -> List[ModelConfig]:
        return list(self._registry.values())

    def get(self, model_name: str) -> ModelConfig:
        config = self._registry.get(model_name)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported model '{model_name}'.",
            )
        return config

    def predict(
        self,
        *,
        model_name: str,
        dataset: pd.DataFrame,
        use_preview: bool,
        return_csv: bool,
        target_column: Optional[str] = None,
    ) -> PredictionResult:
        config = self.get(model_name)
        cleaned_df = self._sanitize_dataset(dataset)
        features_df, target_series, detected_target = self._prepare_features(
            cleaned_df, config, target_column
        )
        pipeline = self._build_pipeline(features_df, config)
        prediction_matrix, prediction_cols = self._run_model(
            pipeline, config, features_df, target_series
        )
        output_df = self._append_predictions(cleaned_df, prediction_matrix, prediction_cols)

        preview_payload = (
            output_df.head(5).to_dict(orient="records") if use_preview else None
        )
        csv_payload = self._build_csv_payload(output_df) if return_csv else None
        predictions_payload: List[Any]
        if len(prediction_cols) == 1:
            predictions_payload = output_df[prediction_cols[0]].tolist()
        else:
            predictions_payload = output_df[prediction_cols].to_dict(orient="records")

        metadata = {
            "rows": len(output_df),
            "columns": list(output_df.columns),
            "task_type": config.task_type,
            "target_column": detected_target,
        }

        return PredictionResult(
            model_name=config.name,
            model_type=config.task_type,
            predictions=predictions_payload,
            metadata=metadata,
            preview=preview_payload,
            csv_base64=csv_payload["content"] if csv_payload else None,
            csv_filename=csv_payload["filename"] if csv_payload else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV contains no rows.",
            )
        cleaned = dataset.dropna(how="all")
        if cleaned.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset only has empty rows after cleaning.",
            )
        return cleaned

    def _prepare_features(
        self,
        dataset: pd.DataFrame,
        config: ModelConfig,
        target_column: Optional[str],
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[str]]:
        working_df = dataset.copy()
        target_series: Optional[pd.Series] = None
        detected_target: Optional[str] = None
        if config.requires_target:
            detected_target = self._detect_target_column(working_df, target_column)
            target_series = working_df.pop(detected_target)
            if target_series.isnull().all():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Target column contains only null values.",
                )
        if working_df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset must include at least one feature column.",
            )
        return working_df, target_series, detected_target

    @staticmethod
    def _detect_target_column(
        dataset: pd.DataFrame, target_column: Optional[str]
    ) -> str:
        if target_column and target_column in dataset.columns:
            return target_column
        candidate_names = [
            target_column,
            "target",
            "label",
            "y",
            "class",
        ]
        for candidate in candidate_names:
            if candidate and candidate in dataset.columns:
                return candidate
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Could not infer target column. Provide 'target_column' in the form data "
                "or include a column named 'target', 'label', 'y', or 'class'."
            ),
        )

    def _build_pipeline(self, dataset: pd.DataFrame, config: ModelConfig) -> Pipeline:
        preprocessor = self._build_preprocessor(dataset)
        estimator = config.estimator_factory()
        if isinstance(estimator, PCA):
            # Keep at most 3 principal components while avoiding shape errors.
            n_components = max(1, min(3, dataset.shape[1]))
            estimator.set_params(n_components=n_components)
        steps: List[Tuple[str, Any]] = []
        if preprocessor is not None:
            steps.append(("prep", preprocessor))
        steps.append(("model", estimator))
        return Pipeline(steps=steps)

    @staticmethod
    def _build_preprocessor(dataset: pd.DataFrame) -> Optional[ColumnTransformer]:
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [
            col for col in dataset.columns if col not in set(numeric_cols)
        ]
        transformers: List[Tuple[str, Any, List[str]]] = []
        if numeric_cols:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_cols,
                )
            )
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    sparse_output=False,
                                ),
                            ),
                        ]
                    ),
                    categorical_cols,
                )
            )
        if not transformers:
            return None
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _run_model(
        self,
        pipeline: Pipeline,
        config: ModelConfig,
        features_df: pd.DataFrame,
        target_series: Optional[pd.Series],
    ) -> Tuple[np.ndarray, List[str]]:
        if config.task_type in {"regression", "classification"}:
            if target_series is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Target data is required for supervised models.",
                )
            pipeline.fit(features_df, target_series)
            predictions = pipeline.predict(features_df)
            return np.asarray(predictions), ["prediction"]

        if config.task_type == "clustering":
            if hasattr(pipeline, "fit_predict"):
                predictions = pipeline.fit_predict(features_df)
            else:
                pipeline.fit(features_df)
                predictions = pipeline.predict(features_df)
            return np.asarray(predictions), ["cluster_label"]

        if config.task_type == "anomaly":
            if hasattr(pipeline, "fit_predict"):
                predictions = pipeline.fit_predict(features_df)
            else:
                pipeline.fit(features_df)
                predictions = pipeline.predict(features_df)
            return np.asarray(predictions), ["anomaly_flag"]

        if config.task_type == "dimensionality_reduction":
            transformed = pipeline.fit_transform(features_df)
            transformed = np.asarray(transformed)
            component_names = [
                f"component_{idx + 1}" for idx in range(transformed.shape[1])
            ]
            return transformed, component_names

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unhandled task type: {config.task_type}",
        )

    @staticmethod
    def _append_predictions(
        dataset: pd.DataFrame, prediction_matrix: np.ndarray, columns: List[str]
    ) -> pd.DataFrame:
        output_df = dataset.copy()
        matrix = prediction_matrix
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        for idx, column in enumerate(columns):
            output_df[column] = matrix[:, idx]
        return output_df

    @staticmethod
    def _build_csv_payload(dataset: pd.DataFrame) -> Dict[str, str]:
        csv_bytes = dataset.to_csv(index=False).encode("utf-8")
        payload = base64.b64encode(csv_bytes).decode("ascii")
        return {"filename": "predictions.csv", "content": payload}

    @staticmethod
    def _build_registry() -> Dict[str, ModelConfig]:
        registry: Dict[str, ModelConfig] = {}

        registry["linear_regression"] = ModelConfig(
            name="linear_regression",
            display_name="Linear Regression",
            version="1.0",
            task_type="regression",
            description="Standard OLS regression with scaling and imputers.",
            estimator_factory=lambda: LinearRegression(),
        )

        registry["logistic_regression"] = ModelConfig(
            name="logistic_regression",
            display_name="Logistic Regression",
            version="1.0",
            task_type="classification",
            description="Binary/multiclass logistic regression with scaling.",
            estimator_factory=lambda: LogisticRegression(max_iter=1000),
        )

        registry["knn_classifier"] = ModelConfig(
            name="knn_classifier",
            display_name="KNN Classifier",
            version="1.0",
            task_type="classification",
            description="K-Nearest Neighbors with k=5 and scaling.",
            estimator_factory=lambda: KNeighborsClassifier(n_neighbors=5),
        )

        registry["svm_classifier"] = ModelConfig(
            name="svm_classifier",
            display_name="SVM Classifier",
            version="1.0",
            task_type="classification",
            description="Support Vector Machine with RBF kernel.",
            estimator_factory=lambda: SVC(probability=True),
        )

        registry["decision_tree"] = ModelConfig(
            name="decision_tree",
            display_name="Decision Tree",
            version="1.0",
            task_type="classification",
            description="Gini-based decision tree classifier.",
            estimator_factory=lambda: DecisionTreeClassifier(random_state=42),
        )

        registry["decision_tree_regressor"] = ModelConfig(
            name="decision_tree_regressor",
            display_name="Decision Tree Regressor",
            version="1.0",
            task_type="regression",
            description="Decision tree regressor with depth auto-tuned by the data.",
            estimator_factory=lambda: DecisionTreeRegressor(random_state=42),
        )

        registry["random_forest_regressor"] = ModelConfig(
            name="random_forest_regressor",
            display_name="Random Forest Regressor",
            version="1.0",
            task_type="regression",
            description="200-tree random forest regressor.",
            estimator_factory=lambda: RandomForestRegressor(
                n_estimators=200, random_state=42
            ),
        )

        registry["naive_bayes"] = ModelConfig(
            name="naive_bayes",
            display_name="Gaussian Naive Bayes",
            version="1.0",
            task_type="classification",
            description="Gaussian Naive Bayes for continuous features.",
            estimator_factory=lambda: GaussianNB(),
        )

        registry["kmeans"] = ModelConfig(
            name="kmeans",
            display_name="KMeans Clustering",
            version="1.0",
            task_type="clustering",
            description="KMeans clustering with 8 clusters.",
            estimator_factory=lambda: KMeans(n_clusters=8, random_state=42, n_init=10),
        )

        registry["dbscan"] = ModelConfig(
            name="dbscan",
            display_name="DBSCAN",
            version="1.0",
            task_type="clustering",
            description="Density-based clustering with default eps/min_samples.",
            estimator_factory=lambda: DBSCAN(),
        )

        registry["pca"] = ModelConfig(
            name="pca",
            display_name="PCA",
            version="1.0",
            task_type="dimensionality_reduction",
            description="Principal Component Analysis capturing top 3 components.",
            estimator_factory=lambda: PCA(n_components=3),
        )

        registry["isolation_forest"] = ModelConfig(
            name="isolation_forest",
            display_name="Isolation Forest",
            version="1.0",
            task_type="anomaly",
            description="Isolation Forest for anomaly detection.",
            estimator_factory=lambda: IsolationForest(random_state=42),
        )

        if XGBClassifier is not None:
            registry["xgboost_classifier"] = ModelConfig(
                name="xgboost_classifier",
                display_name="XGBoost Classifier",
                version="1.0",
                task_type="classification",
                description="Gradient boosted trees via XGBoost.",
                estimator_factory=lambda: XGBClassifier(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    verbosity=0,
                ),
            )

        if XGBRegressor is not None:
            registry["xgboost_regressor"] = ModelConfig(
                name="xgboost_regressor",
                display_name="XGBoost Regressor",
                version="1.0",
                task_type="regression",
                description="Gradient boosted decision tree regressor via XGBoost.",
                estimator_factory=lambda: XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    verbosity=0,
                ),
            )

        return registry


model_registry = ModelRegistry()
