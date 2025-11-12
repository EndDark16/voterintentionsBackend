"""Training entry point for the voter intention KNN classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from custom_transformers import SecondaryChoiceImputer

DATA_PATH = Path("voter_intentions_3000.csv")
ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports")
MODEL_PATH = ARTIFACT_DIR / "knn_voter_intentions.joblib"
METRICS_PATH = REPORT_DIR / "knn_metrics.json"
CLASSIFICATION_REPORT_PATH = REPORT_DIR / "knn_classification_report.json"
CONFUSION_MATRIX_CSV = REPORT_DIR / "knn_confusion_matrix.csv"
CONFUSION_MATRIX_PNG = REPORT_DIR / "knn_confusion_matrix.png"

TARGET_COL = "intended_vote"
NUMERIC_FEATURES: List[str] = [
    "age",
    "gender",
    "education",
    "employment_status",
    "employment_sector",
    "income_bracket",
    "marital_status",
    "household_size",
    "has_children",
    "urbanicity",
    "region",
    "voted_last",
    "party_id_strength",
    "union_member",
    "public_sector",
    "home_owner",
    "small_biz_owner",
    "owns_car",
    "wa_groups",
    "refused_count",
    "attention_check",
    "will_turnout",
    "undecided",
    "preference_strength",
    "survey_confidence",
    "tv_news_hours",
    "social_media_hours",
    "trust_media",
    "civic_participation",
    "job_tenure_years",
]
CAT_FEATURES: List[str] = ["primary_choice", "secondary_choice"]


def ensure_output_dirs() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                IterativeImputer(
                    estimator=LinearRegression(),
                    max_iter=25,
                    random_state=42,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            (
                "secondary_choice_imputer",
                SecondaryChoiceImputer(numeric_features=NUMERIC_FEATURES),
            ),
            ("preprocessor", preprocessor),
            ("knn", KNeighborsClassifier()),
        ]
    )
    return pipeline


def evaluate_and_persist(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test_encoded: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, float]:
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test = label_encoder.inverse_transform(y_test_encoded)

    report_dict = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=200)
    plt.close()

    conf_matrix = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
    conf_matrix.to_csv(CONFUSION_MATRIX_CSV)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with CLASSIFICATION_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    joblib.dump({"model": model, "label_encoder": label_encoder}, MODEL_PATH)
    return metrics


def main() -> None:
    ensure_output_dirs()
    df = load_dataset()

    X = df[NUMERIC_FEATURES + CAT_FEATURES]
    y = df[TARGET_COL]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    pipeline = build_pipeline()

    param_grid = {
        "knn__n_neighbors": [5, 11, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1_macro",
        verbose=2,
    )

    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Cross-val score (f1_macro): {grid.best_score_:.4f}")

    metrics = evaluate_and_persist(grid.best_estimator_, X_test, y_test, label_encoder)
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
