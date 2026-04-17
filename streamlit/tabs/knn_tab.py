# Student: Marvin Adorian Zanchi Santos | C00288302
# ============================================================
# KNN CLASSIFIER TAB — Marathon Performance Classification
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def classify_performance(finish_seconds: float) -> str:
    if finish_seconds < 12600:
        return "Fast"
    elif finish_seconds <= 18000:
        return "Average"
    return "Slow"


@st.cache_resource
def train_knn_model():
    data_paths = [
        "data/Results.csv",
        "../data/Results.csv",
        "../../data/Results.csv",
    ]

    data = None
    for path in data_paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            break

    if data is None:
        return None, None, None, None

    df = data.copy()
    df = df[(df["Age"] > 15) & (df["Age"] < 91) & (df["Age"] != -1)]
    df = df[df["Gender"].isin(["M", "F"])]
    df = df[(df["Finish"] > 0) & (df["Finish"] <= 20000)]
    df = df.drop(columns=["Name", "Year", "Age Bracket"], errors="ignore")

    df["Performance"] = df["Finish"].apply(classify_performance)
    df["Gender_Encoded"] = (df["Gender"] == "M").astype(int)

    top_races = df["Race"].value_counts().head(20).index
    df = df[df["Race"].isin(top_races)].copy()

    race_dummies = pd.get_dummies(df["Race"], prefix="Race", drop_first=True)
    features = pd.concat([df[["Age", "Gender_Encoded"]], race_dummies], axis=1)
    target = df["Performance"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(features_scaled, target)

    feature_columns = list(features.columns)
    race_names = sorted([
        c.replace("Race_", "") for c in feature_columns if c.startswith("Race_")
    ])

    return model, scaler, feature_columns, race_names


def render():
    st.header("📊 KNN Runner Classifier")
    st.markdown(
        "Classifies a runner as **Fast**, **Average**, or **Slow** using the "
        "**K-Nearest Neighbours** algorithm, based on age, gender, and race."
    )

    with st.spinner("Loading dataset and training KNN model (first run only)..."):
        model, scaler, columns, race_names = train_knn_model()

    if model is None:
        st.error(
            "Dataset not found. Place `Results.csv` in the `data/` folder at "
            "the project root."
        )
        return

    # --- Inputs ---
    st.subheader("Your Details")

    age = st.slider("Age", 16, 90, 30, key="knn_age")
    gender_label = st.radio(
        "Gender", ["Male", "Female"], horizontal=True, key="knn_gender"
    )
    gender = 1 if gender_label == "Male" else 0

    race_selected = st.selectbox("Marathon Race", options=race_names, key="knn_race")

    # --- Prediction ---
    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0
    input_data.loc[0, "Age"] = age
    input_data.loc[0, "Gender_Encoded"] = gender

    race_col = f"Race_{race_selected}"
    if race_col in columns:
        input_data.loc[0, race_col] = 1

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    st.markdown("---")
    st.subheader("Classification Result")

    colour_map = {"Fast": "🟢", "Average": "🟠", "Slow": "🔴"}
    st.markdown(f"### {colour_map.get(prediction, '⚪')} Predicted: **{prediction}**")

    # Smaller chart
    fig, ax = plt.subplots(figsize=(4, 2.5))
    bar_colours = [
        "#2196F3" if cls == prediction else "#bdc3c7" for cls in classes
    ]
    bars = ax.bar(classes, probabilities * 100, color=bar_colours)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Vote distribution among the 5 nearest neighbours", fontsize=10)

    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{prob*100:.0f}%",
            ha="center", fontsize=9,
        )

    st.pyplot(fig, use_container_width=False)
    plt.close()

    st.markdown("---")

    with st.expander("ℹ️ KNN vs Linear Regression"):
        st.markdown(
            """
            **Linear Regression** predicts a specific time value — for example,
            "your predicted finish time is 4h 12m".

            **KNN** classifies into a category instead — "you are a Fast
            runner".

            KNN works by finding the 5 most similar runners in the dataset
            (based on age, gender, and race) and voting on the most common
            category among them.

            **Why both approaches are useful:**
            - Linear Regression is better when a precise numeric prediction is
              needed.
            - KNN is better when a general category is enough, and it also
              shows how confident the model is by displaying the vote
              distribution.
            """
        )

    with st.expander("💻 Source Code — KNN Training"):
        st.code("""# Create performance categories from finish time
def classify_performance(finish_seconds):
    if finish_seconds < 12600:    # Under 3.5 hours
        return 'Fast'
    elif finish_seconds <= 18000:  # 3.5 to 5 hours
        return 'Average'
    else:
        return 'Slow'

df['Performance'] = df['Finish'].apply(classify_performance)

# Encode features
df['Gender_Encoded'] = (df['Gender'] == 'M').astype(int)
race_dummies = pd.get_dummies(df['Race'], prefix='Race', drop_first=True)
features = pd.concat([df[['Age', 'Gender_Encoded']], race_dummies], axis=1)

# Scale features (critical for KNN — distance-based)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train KNN and find optimal k
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)

# Final model with best k
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)""", language="python")