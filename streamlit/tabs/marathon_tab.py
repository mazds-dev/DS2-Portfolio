# Student: Marvin Adorian Zanchi Santos | C00288302
# ============================================================
# MARATHON PREDICTOR TAB — Linear Regression (Semester 1)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


@st.cache_resource
def load_marathon_model():
    """Load the trained Linear Regression model from Semester 1."""
    search_paths = [
        "marathon_model.pkl",
        "../marathon_model.pkl",
        "models/marathon_model.pkl",
        "../models/marathon_model.pkl",
    ]
    columns_paths = [
        "model_columns.pkl",
        "../model_columns.pkl",
        "models/model_columns.pkl",
        "../models/model_columns.pkl",
    ]

    model = None
    columns = None

    for path in search_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break

    for path in columns_paths:
        if os.path.exists(path):
            columns = joblib.load(path)
            break

    return model, columns


def format_time(seconds: float) -> str:
    """Convert seconds into a readable hh:mm:ss string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render():
    """Render the Marathon Predictor tab."""
    st.header("🏃 Marathon Finish Time Predictor")
    st.markdown(
        "Predicts marathon finish time using **Linear Regression** based on "
        "age, gender, and race. This was the first algorithm implemented in "
        "the portfolio (Semester 1)."
    )

    model, columns = load_marathon_model()

    if model is None or columns is None:
        st.error(
            "Marathon model not found. Please ensure `marathon_model.pkl` "
            "and `model_columns.pkl` are in the project root or in a "
            "`models/` folder."
        )
        return

    # --- Input form ---
    st.subheader("Your Details")

    age = st.slider("Age", 16, 90, 30, help="Your age in years")

    gender_label = st.radio(
        "Gender", options=["Male", "Female"], horizontal=True
    )
    gender = 1 if gender_label == "Male" else 0

    race_columns = [c for c in columns if c.startswith("Race_")]
    race_names = sorted([c.replace("Race_", "") for c in race_columns])

    race_selected = st.selectbox("Marathon Race", options=race_names)

    # --- Build input ---
    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0
    input_data.loc[0, "Age"] = age
    input_data.loc[0, "Gender"] = gender

    race_col = f"Race_{race_selected}"
    if race_col in columns:
        input_data.loc[0, race_col] = 1

    # --- Prediction ---
    predicted_seconds = float(model.predict(input_data)[0])
    predicted_hours = predicted_seconds / 3600

    st.markdown("---")
    st.subheader("Prediction")

    st.metric(
        label="Predicted finish time",
        value=format_time(predicted_seconds),
    )
    st.caption(f"Approximately {predicted_hours:.2f} hours")

    if predicted_hours < 3.5:
        st.success("🏆 **Fast runner** — elite-level performance.")
    elif predicted_hours <= 5.0:
        st.info("⚡ **Average runner** — solid recreational performance.")
    else:
        st.warning("🐢 **Slower pace** — great for first-time or casual runners.")

    st.markdown("---")

    with st.expander("ℹ️ How the model works"):
        st.markdown(
            """
            This model uses **Linear Regression** — one of the simplest and
            most interpretable machine learning algorithms. It learned from
            ~429,000 real marathon results from 2023.

            **Patterns discovered by the model:**
            - **Age:** Each extra year adds about **34 seconds** to finish time
            - **Gender:** Male runners finish about **19 minutes faster** on average
            - **Race:** Different events have different difficulty due to terrain,
              elevation, and climate

            **Model performance:**
            - R² Score: ~0.19
            - Average error (MAE): ~32 minutes
            """
        )

    with st.expander("💻 Source Code — Model Training"):
        st.code("""# Prepare Features and Target
X = df[['Age', 'Gender', 'Race']]
y = df['Finish']

# One-hot encode the 'Race' column
X = pd.get_dummies(X, columns=['Race'], drop_first=True)

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f}")
print(f"MAE (minutes): {mae/60:.2f}")
print(f"RMSE (minutes): {rmse/60:.2f}")""", language="python")