# ✈️ Flight Price Prediction

A machine learning application that predicts flight ticket prices from flight, route, schedule and booking information.

The project was developed as part of a Kaggle flight price prediction assignment and deployed as an interactive **Streamlit** application.

## 📌 Project Performance

| Metric | Score |
|---|---:|
| Validation R² | **0.98132** |
| Kaggle Leaderboard R² | **0.982** |
| Model | XGBoost Regressor |

The Kaggle evaluation metric is **R² Score**.

---

## 🎯 Problem Statement

Predict the price of a flight ticket based on information such as:

- Airline
- Flight code
- Source city
- Destination city
- Departure time period
- Arrival time period
- Number of stops
- Seat class
- Flight duration
- Number of days left before the journey

---

## 🗂️ Project Structure

```text
flight-price-prediction/
│
├── app.py
├── model_pipeline.py
├── train.csv
├── requirements.txt
└── README.md
```

> **Important:** `train.csv` must be available in the repository because the Streamlit app trains the final model when it starts.

If the dataset is too large for a normal GitHub repository, use Git LFS or replace the startup-training approach with a saved model artifact such as `model.joblib`.

---

## 🧠 Machine Learning Pipeline

The application uses a single Scikit-learn `Pipeline` containing:

```text
Input Data
    ↓
ColumnTransformer
    ├── Numeric features
    │     └── Median imputation
    │
    ├── Nominal categorical features
    │     ├── Most-frequent imputation
    │     └── One-Hot Encoding
    │
    └── Ordinal features
          ├── Most-frequent imputation
          └── Ordinal Encoding
    ↓
XGBoost Regressor
    ↓
Predicted Flight Price
```

### Numeric features

- `duration`
- `days_left`

### One-hot encoded features

- `airline`
- `source`
- `departure`
- `arrival`
- `destination`

### Ordinal encoded features

- `flight`
- `stops`
- `class`

Unknown categories are handled safely during inference.

---

## ⚙️ XGBoost Configuration

The deployed model uses the tuned configuration from the Kaggle solution:

```python
XGBRegressor(
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
    n_jobs=-1
)
```

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd flight-price-prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push these files to GitHub:
   - `app.py`
   - `model_pipeline.py`
   - `train.csv`
   - `requirements.txt`
   - `README.md`

2. Open Streamlit Community Cloud.

3. Connect your GitHub repository.

4. Select:
   - **Main file:** `app.py`

5. Deploy the application.

The app will train the XGBoost model once when the application starts. Streamlit's `st.cache_resource` prevents unnecessary retraining during normal app interactions.

---

## 🧪 Example Prediction

Example input:

```text
Airline: air india
Source: delhi
Destination: mumbai
Departure: morning
Arrival: evening
Flight: AI-202
Stops: one
Class: economy
Duration: 5.0 hours
Days Left: 30
```

The application returns the predicted ticket price in Indian Rupees.

---

## 🔍 Important Data-Preparation Note

The original Kaggle notebook contained this operation:

```python
train[cat_cols] = train[cat_cols].apply(lambda col: col.str.lower())
```

However, `cat_cols` was not defined in the supplied code, and the transformation happened after `X` had already been created.

For deployment, this has been corrected by explicitly defining the categorical columns and applying the cleaning step **before the model receives the data**.

The deployment pipeline therefore uses:

```python
df[col] = df[col].astype("string").str.strip().str.lower()
```

This keeps training and prediction preprocessing consistent.

---

## 📊 Presentation Talking Points

For the project presentation, you can explain the deployment architecture as:

> **User Input → Data Cleaning → Preprocessing Pipeline → XGBoost Model → Predicted Flight Price**

Key points:

- The model achieved **0.98132 R² on validation data**.
- The final Kaggle submission achieved **0.982 R²**.
- Preprocessing is integrated into the ML pipeline.
- The same preprocessing logic is used during prediction.
- `OneHotEncoder(handle_unknown="ignore")` prevents errors from unseen categorical values.
- `OrdinalEncoder` uses `unknown_value=-1` for unseen ordinal categories.
- Streamlit provides an interactive interface for real-time prediction.

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- GitHub

---

## 📄 License

This project is intended for educational and portfolio purposes.
