# Student: Marvin Adorian Zanchi Santos | C00288302
# ============================================================
# K-MEANS EXPLORER TAB — Marathon Runner Clustering
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@st.cache_resource
def train_kmeans_model():
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
    df["Gender_Encoded"] = (df["Gender"] == "M").astype(int)
    df["Finish_Hours"] = df["Finish"] / 3600

    features = df[["Age", "Gender_Encoded", "Finish"]].copy()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(features_scaled)

    cluster_means = df.groupby("Cluster")["Finish"].mean().sort_values()
    labels = ["Fast Runners", "Average Runners", "Slow Runners"]
    label_map = {cluster: labels[i] for i, (cluster, _) in enumerate(cluster_means.items())}
    df["Cluster_Label"] = df["Cluster"].map(label_map)

    return kmeans, scaler, df, label_map


def render():
    st.header("🔍 K-Means Runner Explorer")
    st.markdown(
        "Groups runners into natural clusters using **K-Means** — an "
        "**unsupervised** algorithm that finds structure in data without "
        "being given any labels."
    )

    with st.spinner("Loading data and training K-Means (first run only)..."):
        kmeans, scaler, df, label_map = train_kmeans_model()

    if kmeans is None:
        st.error("Dataset not found. Place `Results.csv` in the `data/` folder.")
        return

    # --- Cluster visualisation ---
    st.subheader("Discovered clusters")

    sample = df.sample(min(3000, len(df)), random_state=42)
    cluster_colours = {
        "Fast Runners": "#2ecc71",
        "Average Runners": "#f39c12",
        "Slow Runners": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for label, colour in cluster_colours.items():
        subset = sample[sample["Cluster_Label"] == label]
        ax.scatter(
            subset["Age"], subset["Finish_Hours"],
            c=colour, alpha=0.3, s=10, label=label,
        )
    ax.set_xlabel("Age")
    ax.set_ylabel("Finish Time (hours)")
    ax.set_title("Marathon Runners — Clusters", fontsize=11)
    ax.legend(markerscale=3, fontsize=9)

    st.pyplot(fig, use_container_width=False)
    plt.close()

    # --- Cluster sizes ---
    st.subheader("Cluster sizes")

    counts = df["Cluster_Label"].value_counts()
    for cluster_name in ["Fast Runners", "Average Runners", "Slow Runners"]:
        count = counts.get(cluster_name, 0)
        percentage = count / len(df) * 100
        colour = cluster_colours[cluster_name]
        st.markdown(
            f"<div style='padding:8px; background-color:{colour}; color:white; "
            f"border-radius:6px; margin-bottom:6px;'>"
            f"<b>{cluster_name}</b> — {count:,} runners ({percentage:.1f}%)"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Interactive ---
    st.subheader("🎯 Find your cluster")
    st.markdown("Enter your details to see which cluster you belong to.")

    your_age = st.slider("Your age", 16, 90, 30, key="kmeans_age")
    your_gender_label = st.radio(
        "Your gender", ["Male", "Female"], horizontal=True, key="kmeans_gender"
    )
    your_gender = 1 if your_gender_label == "Male" else 0
    your_finish_hours = st.slider(
        "Your expected finish time (hours)", 2.0, 6.0, 4.0, 0.1, key="kmeans_finish"
    )
    your_finish_seconds = your_finish_hours * 3600

    your_data = np.array([[your_age, your_gender, your_finish_seconds]])
    your_data_scaled = scaler.transform(your_data)
    your_cluster = kmeans.predict(your_data_scaled)[0]
    your_label = label_map[your_cluster]

    colour = cluster_colours[your_label]
    st.markdown(
        f"<div style='padding:20px; background-color:{colour}; color:white; "
        f"border-radius:10px; text-align:center; font-size:18px; margin-top:10px;'>"
        f"<b>You belong to the {your_label} cluster</b></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    with st.expander("ℹ️ How K-Means is different"):
        st.markdown(
            """
            Unlike KNN (previous tab), K-Means is **unsupervised** — the
            algorithm is not told which runners are fast or slow. It simply
            groups runners by similarity.

            Remarkably, the clusters it discovers closely match the
            categories manually defined for KNN. This shows that the natural
            structure of the data matches intuition about runner performance.

            **Key concepts:**
            - The Elbow Method helps choose the right number of clusters (k=3)
            - Each cluster has a **centroid** — the mean point of all runners
              in that cluster
            - New runners are assigned to the cluster with the nearest centroid
            """
        )

    with st.expander("💻 Source Code — K-Means Clustering"):
        st.code("""# Select and scale features
features = df[['Age', 'Gender_Encoded', 'Finish']].copy()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Elbow Method — find optimal k
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_sample)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(features_sample, labels))

# Fit K-Means with k=3
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans_model.fit_predict(features_scaled)

# Label clusters by mean finish time
cluster_means = df.groupby('Cluster')['Finish'].mean().sort_values()
label_map = {cluster: label for (cluster, _), label in
             zip(cluster_means.items(), ['Fast', 'Average', 'Slow'])}""", language="python")