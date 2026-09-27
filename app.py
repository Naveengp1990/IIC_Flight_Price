"""
Streamlit application for Flight Price Prediction.

Run locally:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from model_pipeline import (
    FEATURE_COLS,
    OHE_COLS,
    ORDINAL_COLS,
    clean_categorical_columns,
    train_model,
)

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "train.csv"


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
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
            padding: 1.2rem;
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


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Training XGBoost model... This may take a moment.")
def get_model():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            "train.csv was not found. Upload/copy the Kaggle training dataset "
            "into the same folder as app.py."
        )

    return train_model(TRAIN_FILE)


@st.cache_data
def get_training_data():
    if not TRAIN_FILE.exists():
        return None

    df = pd.read_csv(TRAIN_FILE)
    return clean_categorical_columns(df)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown(
    '<div class="main-title">✈️ Flight Price Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Predict flight ticket prices using an XGBoost regression model trained
        on airline, route, schedule, stop, class, duration and booking-window features.
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Load model/data
# -------------------------------------------------------------------
try:
    model = get_model()
    training_df = get_training_data()
except Exception as exc:
    st.error(f"Application setup error: {exc}")
    st.stop()


# -------------------------------------------------------------------
# Sidebar - project information
# -------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Information")
    st.write("**Algorithm:** XGBoost Regressor")
    st.write("**Task:** Regression")
    st.write("**Evaluation:** R² Score")
    st.write("**Kaggle R²:** 0.982")
    st.write("**Validation R²:** 0.98132")
    st.divider()
    st.caption(
        "The model uses the same preprocessing and hyperparameters used "
        "for the Kaggle submission."
    )


# -------------------------------------------------------------------
# Input helpers
# -------------------------------------------------------------------
def unique_values(column: str, fallback: list[str]) -> list[str]:
    if training_df is None or column not in training_df.columns:
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


# -------------------------------------------------------------------
# Input form
# -------------------------------------------------------------------
st.subheader("Enter Flight Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox(
            "Airline",
            unique_values(
                "airline",
                ["air india", "indigo", "vistara", "air asia"],
            ),
        )

        source = st.selectbox(
            "Source",
            unique_values(
                "source",
                ["delhi", "mumbai", "bangalore", "chennai"],
            ),
        )

        destination = st.selectbox(
            "Destination",
            unique_values(
                "destination",
                ["mumbai", "delhi", "bangalore", "chennai"],
            ),
        )

        flight = st.text_input(
            "Flight Code",
            value="",
            placeholder="Example: AI-202",
        )

    with col2:
        departure = st.selectbox(
            "Departure Time Period",
            unique_values(
                "departure",
                ["early_morning", "morning", "afternoon", "evening", "night"],
            ),
        )

        arrival = st.selectbox(
            "Arrival Time Period",
            unique_values(
                "arrival",
                ["early_morning", "morning", "afternoon", "evening", "night"],
            ),
        )

        stops = st.selectbox(
            "Number of Stops",
            unique_values(
                "stops",
                ["zero", "one", "two_or_more"],
            ),
        )

        travel_class = st.selectbox(
            "Class",
            unique_values(
                "class",
                ["economy", "business"],
            ),
        )

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


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
if submitted:
    if not flight.strip():
        st.warning("Please enter a flight code.")
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

    input_data = clean_categorical_columns(input_data)
    input_data = input_data[FEATURE_COLS]

    prediction = float(model.predict(input_data)[0])

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
        "Prediction generated successfully using the trained XGBoost pipeline."
    )

    with st.expander("View input used for prediction"):
        st.dataframe(input_data, use_container_width=True)


# -------------------------------------------------------------------
# Project summary
# -------------------------------------------------------------------
st.divider()

st.subheader("Project Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown(
        """
        **Preprocessing**
        - Median imputation for numeric features
        - Most-frequent imputation for categorical features
        - One-hot encoding for nominal categorical variables
        - Ordinal encoding for flight, stops and class
        - Unknown categories handled safely during prediction
        """
    )

with summary_col2:
    st.markdown(
        """
        **Model**
        - XGBoost Regressor
        - 1,000 boosting trees
        - Tuned regularization and sampling parameters
        - Final model trained on the complete training dataset
        - Kaggle leaderboard R²: **0.982**
        """
    )

st.caption(
    "Project: Flight Price Prediction | Built with Python, Scikit-learn, "
    "XGBoost and Streamlit"
)
