# Data Science & Machine Learning 2 — Portfolio

### Student Details
- **Name:** Marvin Adorian Zanchi dos Santos
- **Student Number:** C00288302
- **Course:** BSc Software Development
- **Module:** Data Science & Machine Learning 2
- **Lecturer:** Ben OShaughnessy
- **Submission Date:** 17 April 2026

---

# Portfolio Overview

This portfolio extends the Semester 1 Linear Regression project by implementing five additional machine learning algorithms, covering supervised learning, unsupervised learning, neural networks, reinforcement learning, and deep learning.

It includes an interactive **Streamlit application** that showcases all six algorithms with live demos.

---

# Topics Covered

| # | Topic | Algorithm | Dataset | Notebook |
|---|-------|-----------|---------|---------|
| 0 | Regression (Sem 1) | Linear Regression | Marathon Results | `marathon_predicting_finish_time.ipynb` |
| 1 | Classification | K-Nearest Neighbours (KNN) | Marathon Results | `01_KNN_Marathon_Classification.ipynb` |
| 2 | Clustering | K-Means | Marathon Results | `02_KMeans_Marathon_Clustering.ipynb` |
| 3 | Neural Networks | ANN (Dense layers) | MNIST Handwritten Digits | `03_Neural_Network_Digit_Recognition.ipynb` |
| 4 | Reinforcement Learning | Q-Learning | Custom Gridworld | `04_Reinforcement_Learning_Gridworld.ipynb` |
| 5 | Deep Learning | CNN | Fashion MNIST | `05_CNN_Fashion_Classification.ipynb` |

---

# Datasets

### Marathon Results (Topics 1 & 2)
- **Source:** [Kaggle – 2023 Marathon Results](https://www.kaggle.com/datasets/runningwithrock/2023-marathon-results?resource=download&select=Results.csv)
- **Size:** ~429,000 entries from 600+ events across the United States
- **Features:** Age, Gender, Race, Finish time (seconds)
- **Note:** Due to its size, the dataset is **not included** in this repository. Download it from Kaggle and place it inside the `/data` folder.

### MNIST Handwritten Digits (Topic 3)
- **Source:** Built into Keras/TensorFlow (LeCun, Cortes & Burges, 1998)
- **Size:** 70,000 grayscale images (28×28 pixels), 10 classes (digits 0–9)

### Custom Gridworld (Topic 4)
- **Source:** Based on module course materials, extended with a 6×6 grid and wall obstacles
- **Description:** Grid environment where a Q-Learning agent learns to navigate from start to goal

### Fashion MNIST (Topic 5)
- **Source:** Built into Keras/TensorFlow (Xiao, Rasul & Vollgraf, 2017)
- **Size:** 70,000 grayscale images (28×28 pixels), 10 clothing categories

---

# Tools & Technologies

- **Language:** Python 3.12
- **ML Libraries:** scikit-learn, TensorFlow/Keras
- **Data Libraries:** pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Deployment:** Streamlit, streamlit-drawable-canvas
- **Environment:** Jupyter Notebook / Visual Studio Code
- **Version Control:** GitHub

---

# Methodology

### Topic 1 — KNN (Marathon Runner Classification)
- Created performance categories: **Fast** (< 3.5h), **Average** (3.5–5h), **Slow** (> 5h)
- Encoded features: Age, Gender, Race (one-hot)
- Scaled features using StandardScaler (critical for distance-based algorithms)
- Baseline model with k=5, then hyperparameter tuning (k=1 to k=30)
- Evaluated with confusion matrix and classification report

### Topic 2 — K-Means (Marathon Runner Clustering)
- Applied Elbow Method and Silhouette Score to determine optimal k=3
- Discovered natural clusters without labels (unsupervised)
- Compared clusters with KNN categories to illustrate supervised vs unsupervised learning

### Topic 3 — ANN (Handwritten Digit Recognition)
- Architecture: 784 → 128 (ReLU) → Dropout → 64 (ReLU) → Dropout → 10 (Softmax)
- Trained for 20 epochs on 60,000 MNIST images
- Test accuracy: **~97%**
- Deployed as interactive Streamlit canvas where users draw digits in real time

### Topic 4 — Q-Learning (Reinforcement Learning)
- Custom 6×6 Gridworld with 6 wall obstacles (extended from course materials)
- Implemented RandomAgent baseline and Q-Learning agent with epsilon-greedy exploration
- Trained for 500 episodes, analysed 5 hyperparameter configurations (α, γ, ε)
- Visualised Q-value heatmap and learned policy as arrows

### Topic 5 — CNN (Fashion Item Classification)
- Architecture: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(10)
- Compared CNN vs ANN baseline on Fashion MNIST
- CNN test accuracy: **~90%** vs ANN: **~87%**
- Visualised learned convolutional filters from the first layer

---

# Key Results

| Algorithm | Dataset | Metric | Result |
|-----------|---------|--------|--------|
| Linear Regression (Sem 1) | Marathon | R² | 0.19 |
| KNN | Marathon | Accuracy | ~75% |
| K-Means | Marathon | Silhouette Score | ~0.45 |
| ANN | MNIST Digits | Accuracy | ~97% |
| Q-Learning | Gridworld | Final avg reward | ~–0.5 |
| CNN | Fashion MNIST | Accuracy | ~90% |

---

# Interactive Streamlit Application

The portfolio includes a unified Streamlit app with six interactive tabs — one per algorithm.

### How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the marathon dataset here:
   ```
   /data/Results.csv
   ```

3. Generate the trained models by running the notebooks first (in order: 03, 04, 05).

4. Run the app:
   ```bash
   cd streamlit
   streamlit run app.py
   ```

5. Open your browser at `http://localhost:8501`

> **Note:** If the drawable canvas does not appear in the Digits tab, close and reopen the browser. This is a known behaviour of the `streamlit-drawable-canvas` library.

### App Tabs

| Tab | Algorithm | Interaction |
|-----|-----------|-------------|
| 🏃 Marathon | Linear Regression | Input age, gender, race → predict finish time |
| 📊 KNN | K-Nearest Neighbours | Input details → classify as Fast/Average/Slow |
| 🔍 K-Means | Clustering | Explore clusters → find your cluster |
| ✍️ Digits | Neural Network | Draw a digit on the canvas → real-time recognition |
| 🎮 Gridworld | Q-Learning | Adjust hyperparameters → watch agent learn live |
| 👕 Fashion | CNN | Random sample or upload image → classify item |

---

# Project Structure

```
DS2-Portfolio/
├── data/
│   └── Results.csv              ← (not included, download from Kaggle)
├── notebooks/
│   ├── 00_Portfolio_Outline.ipynb
│   ├── 01_KNN_Marathon_Classification.ipynb
│   ├── 02_KMeans_Marathon_Clustering.ipynb
│   ├── 03_Neural_Network_Digit_Recognition.ipynb
│   ├── 04_Reinforcement_Learning_Gridworld.ipynb
│   └── 05_CNN_Fashion_Classification.ipynb
├── streamlit/
│   ├── app.py
│   └── tabs/
│       ├── marathon_tab.py
│       ├── knn_tab.py
│       ├── kmeans_tab.py
│       ├── digit_tab.py
│       ├── gridworld_tab.py
│       └── fashion_tab.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

# References

- **Dataset (Marathon):** Kaggle – 2023 Marathon Results
- **Dataset (Digits):** MNIST — LeCun, Y., Cortes, C., & Burges, C. (1998)
- **Dataset (Fashion):** Fashion MNIST — Xiao, H., Rasul, K., & Vollgraf, R. (2017)
- **Textbook:** Sutton, R. S., & Barto, A. G. (2020). *Reinforcement Learning: An Introduction* (2nd ed.)
- **Libraries:** scikit-learn, TensorFlow, Keras, pandas, matplotlib, Streamlit
- **Course Materials:** Data Science & Machine Learning 2 — module notes and example notebooks
- **AI Assistance:** Claude (Anthropic) — used for documentation, code structure, and debugging

---

# Author

Developed by **Marvin Adorian Zanchi dos Santos**  
BSc in Software Development, 4th-year student at South East Technological University - Carlow Campus
