"""
Flight Price Prediction - Training Pipeline

Run this file once to:
1. Fit the preprocessing pipeline on train.csv
2. Save the fitted preprocessing pipeline as process.pkl
3. Train XGBoost on the transformed data
4. Save the trained XGBoost model as xgb_model.pkl

After these files are created, Streamlit does NOT retrain the model.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent

TRAIN_FILE = BASE_DIR / "train.csv"
PROCESS_FILE = BASE_DIR / "process.pkl"
MODEL_FILE = BASE_DIR / "xgb_model.pkl"


NUMERIC_COLS = [
    "duration",
    "days_left",
]

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
    Clean categorical columns and convert pandas.NA to numpy.nan.

    This prevents:
        boolean value of NA is ambiguous
    """

    df = df.copy()

    categorical_cols = OHE_COLS + ORDINAL_COLS

    for col in categorical_cols:

        if col not in df.columns:
            continue

        # Use object dtype instead of pandas StringDtype.
        df[col] = df[col].astype(object)

        # Convert missing values to numpy.nan.
        df[col] = df[col].where(
            pd.notna(df[col]),
            np.nan
        )

        # Lowercase only valid string values.
        df[col] = df[col].map(
            lambda value:
                value.strip().lower()
                if isinstance(value, str)
                else value
        )

        # Final missing-value normalization.
        df[col] = df[col].where(
            pd.notna(df[col]),
            np.nan
        )

    return df


def create_preprocessor():
    """
    Create the preprocessing pipeline.
    """

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "num",
                SimpleImputer(
                    strategy="median"
                ),
                NUMERIC_COLS,
            ),

            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),

                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
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
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
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

    return preprocessor


def create_xgb_model():
    """
    XGBoost configuration used for the Kaggle solution.
    """

    return XGBRegressor(
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


def train_and_save():
    """
    Train preprocessing + XGBoost and save both separately.
    """

    print("=" * 60)
    print("FLIGHT PRICE MODEL TRAINING")
    print("=" * 60)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {TRAIN_FILE}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    train = pd.read_csv(TRAIN_FILE)

    print(f"Training data shape: {train.shape}")

    # --------------------------------------------------------
    # Clean categorical columns
    # --------------------------------------------------------
    train = clean_categorical_columns(train)

    X = train[FEATURE_COLS].copy()
    y = train[TARGET_COL].copy()

    # --------------------------------------------------------
    # Fit preprocessing
    # --------------------------------------------------------
    print("\nFitting preprocessing pipeline...")

    preprocessor = create_preprocessor()

    X_processed = preprocessor.fit_transform(X)

    print(
        "Processed feature shape:",
        X_processed.shape
    )

    # --------------------------------------------------------
    # Save fitted preprocessing
    # --------------------------------------------------------
    with open(PROCESS_FILE, "wb") as file:
        pickle.dump(
            preprocessor,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"Saved preprocessing pipeline → {PROCESS_FILE.name}"
    )

    # --------------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------------
    print("\nTraining XGBoost model...")

    model = create_xgb_model()

    model.fit(
        X_processed,
        y
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(
            model,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"Saved trained model → {MODEL_FILE.name}"
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save()
