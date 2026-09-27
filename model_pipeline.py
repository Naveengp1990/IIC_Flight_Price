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


# ============================================================
# FEATURES
# ============================================================

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

FEATURE_COLS = (
    NUMERIC_COLS
    + OHE_COLS
    + ORDINAL_COLS
)

TARGET_COL = "price"


# ============================================================
# DATA CLEANING
# ============================================================

def clean_categorical_columns(df):

    df = df.copy()

    categorical_cols = (
        OHE_COLS
        + ORDINAL_COLS
    )

    for col in categorical_cols:

        if col not in df.columns:
            continue

        # Convert to normal object dtype
        df[col] = df[col].astype(object)

        # Convert pandas NA to numpy NaN
        df[col] = df[col].where(
            pd.notna(df[col]),
            np.nan
        )

        # Lowercase valid strings
        df[col] = df[col].map(
            lambda value:
                value.strip().lower()
                if isinstance(value, str)
                else value
        )

        # Final NA conversion
        df[col] = df[col].where(
            pd.notna(df[col]),
            np.nan
        )

    return df


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():

    preprocessor = ColumnTransformer(

        transformers=[

            # -----------------------------
            # Numerical
            # -----------------------------

            (
                "num",

                SimpleImputer(
                    strategy="median"
                ),

                NUMERIC_COLS
            ),

            # -----------------------------
            # One-Hot Categorical
            # -----------------------------

            (
                "cat",

                Pipeline(
                    steps=[

                        (
                            "imputer",

                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),

                        (
                            "encoder",

                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        )
                    ]
                ),

                OHE_COLS
            ),

            # -----------------------------
            # Ordinal
            # -----------------------------

            (
                "ord",

                Pipeline(
                    steps=[

                        (
                            "imputer",

                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),

                        (
                            "encoder",

                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1
                            )
                        )
                    ]
                ),

                ORDINAL_COLS
            )
        ]
    )

    return preprocessor


# ============================================================
# XGBOOST MODEL
# ============================================================

def create_xgb_model():

    model = XGBRegressor(

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

        objective="reg:squarederror"
    )

    return model


# ============================================================
# TRAIN + SAVE
# ============================================================

def train_and_save():

    print("\n" + "=" * 60)

    print(
        "FLIGHT PRICE PREDICTION "
        "MODEL TRAINING"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Check train.csv
    # --------------------------------------------------------

    if not TRAIN_FILE.exists():

        raise FileNotFoundError(
            f"\ntrain.csv not found at:\n{TRAIN_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading training data...")

    train = pd.read_csv(
        TRAIN_FILE
    )

    print(
        "Training data shape:",
        train.shape
    )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    train = clean_categorical_columns(
        train
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X = train[
        FEATURE_COLS
    ].copy()

    y = train[
        TARGET_COL
    ].copy()

    print(
        "Features:",
        X.shape
    )

    print(
        "Target:",
        y.shape
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    print(
        "\nFitting preprocessing pipeline..."
    )

    preprocessor = create_preprocessor()

    X_processed = (
        preprocessor.fit_transform(X)
    )

    print(
        "Processed data shape:",
        X_processed.shape
    )

    # --------------------------------------------------------
    # SAVE PREPROCESSOR
    # --------------------------------------------------------

    with open(
        PROCESS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            preprocessor,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        "\nSaved:",
        PROCESS_FILE
    )

    # ========================================================
    # TRAIN XGBOOST
    # ========================================================

    print(
        "\nTraining XGBoost model..."
    )

    model = create_xgb_model()

    model.fit(
        X_processed,
        y
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    with open(
        MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print(
        "\nSaved:",
        MODEL_FILE
    )

    print("\n" + "=" * 60)

    print(
        "TRAINING COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)

    print("\nGenerated files:")

    print(
        f"1. {PROCESS_FILE.name}"
    )

    print(
        f"2. {MODEL_FILE.name}"
    )


# ============================================================
# ALLOW DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    train_and_save()

from model_pipeline import train_and_save


if __name__ == "__main__":

    train_and_save()
