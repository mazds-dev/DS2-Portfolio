# Data Science & Machine Learning 2 — Portfolio
**Student:** Marvin Adorian Zanchi Santos  
**Student Number:** C00288302  
**Module:** Data Science & Machine Learning 2  
**Submission Date:** 27 March 2026 (Preliminary)

---

## Project Structure

```
DS2-Portfolio/
├── data/
│   └── Results.csv              ← (not included)
├── notebooks/
│   ├── 00_Portfolio_Outline.ipynb
│   ├── 01_KNN_Marathon_Classification.ipynb
│   ├── 02_KMeans_Marathon_Clustering.ipynb
│   └── 03_Neural_Network_Digit_Recognition.ipynb
├── streamlit/
│   └── streamlit_digit_recogniser.py
├── .gitignore
└── README.md
```

---

## How to Run

### Step 1: Download the Dataset

The marathon dataset is not included in this repository due to its size (~429,000 rows).

1. Download from Kaggle: [2023 Marathon Results](https://www.kaggle.com/datasets)
2. Place the file `Results.csv` inside the `data/` folder

### Step 2: Set Up the Environment

```bash
"C:\Program Files\Python312\python.exe" -m venv ds2-env
ds2-env\Scripts\activate.bat
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel tensorflow streamlit streamlit-drawable-canvas pillow
```

### Step 3: Run the Notebooks

Open each notebook in VSCode or Jupyter, select the `ds2-env` kernel, and run all cells:

1. `01_KNN_Marathon_Classification.ipynb` — requires `data/Results.csv`
2. `02_KMeans_Marathon_Clustering.ipynb` — requires `data/Results.csv`
3. `03_Neural_Network_Digit_Recognition.ipynb` — downloads MNIST automatically

**Note:** Notebook 03 generates the file `digit_recognition_model.keras` which is required by the Streamlit app. The app will automatically search for this file in the `notebooks/` folder.

### Step 4: Run the Streamlit App (Optional)

```bash
cd streamlit
streamlit run streamlit_digit_recogniser.py
```

Draw a digit (0–9) on the canvas and the neural network will recognise it in real time.

---

## Topics Covered

| # | Topic | Algorithm | Dataset | Status |
|---|-------|-----------|---------|--------|
| 0 | Regression (Sem 1) | Linear Regression | Marathon Results | ✅ Complete |
| 1 | Classification | KNN | Marathon Results | ✅ Complete |
| 2 | Clustering | K-Means | Marathon Results | ✅ Complete |
| 3 | Neural Networks | ANN | MNIST Digits | ✅ Complete |
| 4 | Deep Learning | CNN | Fashion MNIST | 📋 Planned |
| 5 | Reinforcement Learning | Q-Learning | Gridworld | 📋 Planned |

---

## Tools & Technologies

- **Language:** Python 3.12
- **ML Libraries:** scikit-learn, TensorFlow/Keras
- **Data Libraries:** pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Deployment:** Streamlit, streamlit-drawable-canvas
- **Environment:** Jupyter Notebook / VSCode
- **Version Control:** GitHub

---

## References

- **Dataset (Marathon):** Kaggle – 2023 Marathon Results
- **Dataset (Digits):** MNIST (LeCun, Cortes & Burges, 1998)
- **Libraries:** scikit-learn, TensorFlow, Keras, pandas, matplotlib, Streamlit
- **Course Materials:** Data Science & Machine Learning 2 module notes
- **AI Assistance:** Claude (Anthropic) — documentation, code structure, and debugging