"""
Flight Price Prediction - Model Pipeline

This module contains the same preprocessing and XGBoost model configuration
used for the Kaggle solution, packaged for Streamlit deployment.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor


NUMERIC_COLS = ["duration", "days_left"]

OHE_COLS = [
    "airline",
    "source",
    "departure",
    "arrival",
    "destination",
]

ORDINAL_COLS = [
    "flight",
    "stops",
    "class",
]

FEATURE_COLS = NUMERIC_COLS + OHE_COLS + ORDINAL_COLS
TARGET_COL = "price"


def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean categorical columns while preserving missing values as np.nan.

    Using pandas StringDtype can create pandas.NA values. Some versions of
    scikit-learn/XGBoost can encounter:
        TypeError: boolean value of NA is ambiguous

    Therefore, categorical columns are converted to object dtype and missing
    values are normalized to numpy.nan before the sklearn pipeline receives them.
    """
    df = df.copy()

    categorical_cols = OHE_COLS + ORDINAL_COLS

    for col in categorical_cols:
        if col in df.columns:
            # Convert to ordinary object dtype rather than pandas StringDtype.
            df[col] = df[col].astype(object)

            # Normalize pandas.NA / None / NaN to np.nan.
            df[col] = df[col].where(pd.notna(df[col]), np.nan)

            # Lowercase only non-missing values.
            df[col] = df[col].map(
                lambda value: value.strip().lower()
                if isinstance(value, str)
                else value
            )

            # Final safety conversion for any remaining pandas missing marker.
            df[col] = df[col].where(pd.notna(df[col]), np.nan)

    return df


def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline used by the Kaggle model."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUMERIC_COLS,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                OHE_COLS,
            ),
            (
                "ord",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ORDINAL_COLS,
            ),
        ]
    )


def create_model() -> Pipeline:
    """Create the complete preprocessing + XGBoost pipeline."""
    xgb_model = XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.06326317249374157,
        subsample=0.9676369335371923,
        colsample_bytree=0.9259356729288222,
        reg_alpha=5.536933475625916,
        reg_lambda=1.7785155532773833,
        min_child_weight=2,
        gamma=3.125108330139476,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("xgb", xgb_model),
        ]
    )


def load_training_data(data_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare training data."""
    df = pd.read_csv(data_path)
    df = clean_categorical_columns(df)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    return X, y


def train_model(data_path: str | Path) -> Pipeline:
    """Train the final model on the complete training dataset."""
    X, y = load_training_data(data_path)

    model = create_model()
    model.fit(X, y)

    return model
