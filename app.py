"""
Flight Price Prediction - Streamlit Application

IMPORTANT:
This application DOES NOT train the model.

It loads:
    process.pkl
    xgb_model.pkl

and generates predictions instantly.
"""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

from model_pipeline import (
    FEATURE_COLS,
    OHE_COLS,
    ORDINAL_COLS,
    clean_categorical_columns,
)


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

PROCESS_FILE = BASE_DIR / "process.pkl"
MODEL_FILE = BASE_DIR / "xgb_model.pkl"


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #d9d9d9;
        background-color: #f8f9fa;
        text-align: center;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load saved artifacts
# ============================================================

@st.cache_resource
def load_artifacts():

    if not PROCESS_FILE.exists():
        raise FileNotFoundError(
            "process.pkl was not found. "
            "Run `python train_model.py` first."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "xgb_model.pkl was not found. "
            "Run `python train_model.py` first."
        )

    with open(PROCESS_FILE, "rb") as file:
        preprocessor = pickle.load(file)

    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)

    return preprocessor, model


@st.cache_data
def load_training_data():

    train_file = BASE_DIR / "train.csv"

    if not train_file.exists():
        return None

    df = pd.read_csv(train_file)

    return clean_categorical_columns(df)


try:

    preprocessor, model = load_artifacts()
    training_df = load_training_data()

except Exception as error:

    st.error(
        f"Application setup error: {error}"
    )

    st.stop()


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">✈️ Flight Price Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Predict flight ticket prices using a trained XGBoost regression model.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("📊 Model Information")

    st.write(
        "**Algorithm:** XGBoost Regressor"
    )

    st.write(
        "**Task:** Regression"
    )

    st.write(
        "**Validation R²:** 0.98132"
    )

    st.write(
        "**Kaggle R²:** 0.982"
    )

    st.write(
        "**Inference:** Pre-trained model"
    )

    st.divider()

    st.success(
        "Model loaded from pickle files. "
        "No training is performed during prediction."
    )


# ============================================================
# Helper
# ============================================================

def unique_values(
    column,
    fallback
):

    if (
        training_df is None
        or column not in training_df.columns
    ):
        return fallback

    values = (
        training_df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    return sorted(values)


# ============================================================
# User Input
# ============================================================

st.subheader("Enter Flight Details")

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Column 1
    # --------------------------------------------------------

    with col1:

        airline = st.selectbox(
            "Airline",
            unique_values(
                "airline",
                [
                    "air india",
                    "indigo",
                    "vistara",
                    "air asia",
                ],
            ),
        )

        source = st.selectbox(
            "Source",
            unique_values(
                "source",
                [
                    "delhi",
                    "mumbai",
                    "bangalore",
                    "chennai",
                ],
            ),
        )

        destination = st.selectbox(
            "Destination",
            unique_values(
                "destination",
                [
                    "mumbai",
                    "delhi",
                    "bangalore",
                    "chennai",
                ],
            ),
        )

        flight = st.text_input(
            "Flight Code",
            placeholder="Example: AI-202",
        )

    # --------------------------------------------------------
    # Column 2
    # --------------------------------------------------------

    with col2:

        departure = st.selectbox(
            "Departure Time Period",
            unique_values(
                "departure",
                [
                    "early_morning",
                    "morning",
                    "afternoon",
                    "evening",
                    "night",
                ],
            ),
        )

        arrival = st.selectbox(
            "Arrival Time Period",
            unique_values(
                "arrival",
                [
                    "early_morning",
                    "morning",
                    "afternoon",
                    "evening",
                    "night",
                ],
            ),
        )

        stops = st.selectbox(
            "Number of Stops",
            unique_values(
                "stops",
                [
                    "zero",
                    "one",
                    "two_or_more",
                ],
            ),
        )

        travel_class = st.selectbox(
            "Class",
            unique_values(
                "class",
                [
                    "economy",
                    "business",
                ],
            ),
        )

    # --------------------------------------------------------
    # Column 3
    # --------------------------------------------------------

    with col3:

        duration = st.number_input(
            "Duration (hours)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.1,
        )

        days_left = st.number_input(
            "Days Left Until Journey",
            min_value=0,
            max_value=365,
            value=30,
            step=1,
        )

    submitted = st.form_submit_button(
        "🔮 Predict Flight Price",
        use_container_width=True,
    )


# ============================================================
# Prediction
# ============================================================

if submitted:

    if not flight.strip():

        st.warning(
            "Please enter a flight code."
        )

        st.stop()

    input_data = pd.DataFrame(
        [
            {
                "airline": airline,
                "source": source,
                "departure": departure,
                "arrival": arrival,
                "destination": destination,
                "flight": flight.strip(),
                "stops": stops,
                "class": travel_class,
                "duration": duration,
                "days_left": days_left,
            }
        ]
    )

    # Same cleaning used during training.
    input_data = clean_categorical_columns(
        input_data
    )

    input_data = input_data[
        FEATURE_COLS
    ]

    # --------------------------------------------------------
    # Preprocessing only
    # --------------------------------------------------------

    X_input = preprocessor.transform(
        input_data
    )

    # --------------------------------------------------------
    # Instant prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X_input
    )[0]

    prediction = float(prediction)

    st.markdown(
        f"""
        <div class="prediction-box">

            <h3>Predicted Flight Price</h3>

            <h1>₹ {prediction:,.2f}</h1>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.success(
        "Prediction generated successfully "
        "using the pre-trained XGBoost model."
    )

    with st.expander(
        "View input used for prediction"
    ):

        st.dataframe(
            input_data,
            use_container_width=True,
        )


# ============================================================
# Project Summary
# ============================================================

st.divider()

st.subheader("Project Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:

    st.markdown(
        """
        **Preprocessing**

        - Median imputation for numerical features
        - Most-frequent imputation for categorical features
        - One-hot encoding for nominal features
        - Ordinal encoding for ordinal features
        - Unknown categories handled safely
        """
    )

with summary_col2:

    st.markdown(
        """
        **Model**

        - XGBoost Regressor
        - 1,000 estimators
        - Tuned hyperparameters
        - Trained once offline
        - Saved as `xgb_model.pkl`
        - Validation R²: **0.98132**
        - Kaggle R²: **0.982**
        """
    )

st.caption(
    "Flight Price Prediction | "
    "Python + Scikit-learn + XGBoost + Streamlit"
)
