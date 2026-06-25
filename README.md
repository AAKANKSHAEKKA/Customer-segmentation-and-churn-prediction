# 📊 Customer Segmentation & Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A complete end-to-end machine learning project that segments customers using unsupervised clustering and predicts churn risk using supervised classification — helping businesses retain high-value customers before they leave.

---

## 🎯 Project Goals

- **Segment** customers into meaningful groups using K-Means & RFM analysis
- **Predict** which customers are likely to churn (Random Forest + XGBoost)
- **Explain** model decisions using SHAP values
- **Visualize** insights with interactive dashboards

---

## 🗂️ Project Structure

```
customer-churn-project/
├── data/
│   ├── raw/                  # Original datasets (not tracked by Git)
│   └── processed/            # Cleaned & feature-engineered data
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_Segmentation.ipynb # Customer Segmentation (RFM + K-Means)
│   └── 03_Churn_Model.ipynb  # Churn Prediction & Evaluation
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py # Cleaning & feature engineering
│   ├── segmentation.py       # RFM + clustering logic
│   ├── churn_model.py        # Model training & prediction
│   └── visualizations.py     # Plotting utilities
├── models/                   # Saved model files (.pkl)
├── reports/
│   └── figures/              # Output plots & charts
├── tests/
│   └── test_pipeline.py      # Unit tests
├── requirements.txt
├── config.yaml
└── main.py                   # Run full pipeline
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-project.git
cd customer-churn-project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline
```bash
python main.py
```

Or explore step by step in Jupyter:
```bash
jupyter notebook notebooks/
```

---

## 📦 Dataset

This project uses the **[Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)** from Kaggle.

Download it and place it at `data/raw/telco_churn.csv`.

Key features:
- Demographics (gender, age, senior citizen)
- Services subscribed (internet, phone, streaming)
- Account info (tenure, contract type, monthly charges)
- **Target:** `Churn` (Yes/No)

---

## 🔬 Methodology

### Step 1 — Customer Segmentation
- **RFM Analysis**: Recency, Frequency, Monetary scoring
- **K-Means Clustering**: Optimal k via Elbow + Silhouette methods
- **Segments identified**: Champions, Loyal, At-Risk, Lost Customers

### Step 2 — Churn Prediction
- Feature engineering (tenure buckets, charge ratios, service counts)
- Models trained: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning via GridSearchCV
- Evaluation: ROC-AUC, Precision-Recall, Confusion Matrix

### Step 3 — Explainability
- SHAP summary plots to understand feature importance
- Per-customer churn risk scores with top contributing factors

---

## 📈 Results

| Model               | ROC-AUC | Precision | Recall | F1-Score |
|---------------------|---------|-----------|--------|----------|
| Logistic Regression | 0.83    | 0.67      | 0.71   | 0.69     |
| Random Forest       | 0.89    | 0.74      | 0.76   | 0.75     |
| **XGBoost**         | **0.91**| **0.78**  | **0.79**| **0.78**|

> XGBoost is the best-performing model and is saved as the production model.

---

## 🛠️ Tech Stack

| Category        | Tools                                    |
|-----------------|------------------------------------------|
| Data Processing | pandas, numpy                            |
| ML Models       | scikit-learn, xgboost                    |
| Explainability  | shap                                     |
| Visualization   | matplotlib, seaborn, plotly              |
| Notebook        | jupyter                                  |
| Config          | PyYAML                                   |

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Your Name**  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

> ⭐ If you find this project useful, please consider giving it a star!
