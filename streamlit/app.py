# ============================================================
# DS2 PORTFOLIO — Interactive Streamlit Application
# Student: Marvin Adorian Zanchi Santos | C00288302
# ============================================================

import streamlit as st

# Import tab modules
from tabs import (
    marathon_tab,
    knn_tab,
    kmeans_tab,
    digit_tab,
    gridworld_tab,
    fashion_tab,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="DS2 Portfolio — Marvin",
    page_icon="🎓",
    layout="centered",  # Changed from "wide" to "centered" for Medium-style narrow layout
    initial_sidebar_state="expanded",
)

# --- Custom CSS for bigger tabs and better layout ---
st.markdown("""
<style>
    /* Constrain main content width (Medium-style) */
    .main .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Bigger, bolder tab labels — no wrapping */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        flex-wrap: nowrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 14px !important;
        height: auto !important;
        white-space: nowrap !important;
    }
    
    .stTabs [data-baseweb="tab"] p {
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    /* Active tab styling */
    .stTabs [aria-selected="true"] {
        background-color: rgba(33, 150, 243, 0.1);
        border-radius: 8px 8px 0 0;
    }
    
    /* Make headers slightly smaller to fit narrower layout */
    h1 {
        font-size: 2rem !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
    }
    
    /* Control image/chart sizes - prevent them from stretching full width */
    .stPyplot > div {
        display: flex;
        justify-content: center;
    }
    
    /* Make buttons full-width inside columns look cleaner */
    .stButton button {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("🎓 DS2 Portfolio")
    st.markdown("**Marvin Adorian Zanchi Santos**")
    st.markdown("**Student Number:** C00288302")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Interactive showcase of six machine learning algorithms implemented "
        "for the Data Science & Machine Learning portfolio."
    )
    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown(
        "Use the tabs to explore each algorithm. Try changing inputs to see "
        "how the models respond in real time."
    )
    st.markdown("---")
    st.markdown("[📂 Source code on GitHub](https://github.com/mazds-dev/DS2-Portfolio)")

# --- Main Title ---
st.title("Data Science & Machine Learning")
st.markdown(
    "Interactive showcase of six machine learning algorithms: "
    "**Linear Regression**, **KNN**, **K-Means**, **Neural Networks**, "
    "**Reinforcement Learning**, and **Convolutional Neural Networks**."
)

st.markdown("")  # spacing

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏃 Marathon",
    "📊 KNN",
    "🔍 K-Means",
    "✍️ Digits",
    "🎮 Gridworld",
    "👕 Fashion",
])

with tab1:
    marathon_tab.render()

with tab2:
    knn_tab.render()

with tab3:
    kmeans_tab.render()

with tab4:
    digit_tab.render()

with tab5:
    gridworld_tab.render()

with tab6:
    fashion_tab.render()