# 🛡️ Credit Card Fraud Detection — AI System

A complete machine learning pipeline for detecting fraudulent credit card transactions, featuring **4 competing ML models**, interactive **SHAP explainability**, a **premium web dashboard**, and production-ready model deployment.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-purple)

---

## 📋 Overview

This project tackles the real-world challenge of credit card fraud detection using the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, only 0.17% fraudulent).

### Key Features
- 🔬 **Extensive EDA** with premium visualizations
- ⚖️ **SMOTE** for handling extreme class imbalance
- 🤖 **4 ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- 📊 **Model comparison** with ROC/PR curves
- 🧠 **SHAP explainability** for understanding predictions
- 🌐 **Flask web dashboard** for real-time fraud detection
- 📦 **Production-ready** model persistence with joblib

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 4. Train Models
```bash
python train_model.py
```

### 5. Launch Web Dashboard
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 📁 Project Structure
```
credit-card-fraud-detection/
├── main.ipynb          # Enhanced Jupyter notebook (full analysis)
├── train_model.py      # Standalone model training script
├── app.py              # Flask web application
├── requirements.txt    # Python dependencies
├── creditcard.csv      # Dataset (download from Kaggle)
├── models/             # Saved model artifacts
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── metadata.json
├── templates/
│   └── index.html      # Web dashboard template
├── static/
│   ├── css/style.css   # Premium dark theme styles
│   └── js/app.js       # Frontend JavaScript
├── .gitignore
└── README.md
```

---

## 🤖 Models & Results

| Model | AUC-ROC | PR AUC | F1 Score | Fraud Recall |
|-------|---------|--------|----------|-------------|
| Logistic Regression | ~0.97 | ~0.72 | ~0.11 | ~0.92 |
| Random Forest | ~0.97 | ~0.84 | ~0.86 | ~0.80 |
| XGBoost | ~0.98 | ~0.85 | ~0.87 | ~0.82 |
| **LightGBM** ★ | **~0.98** | **~0.86** | **~0.88** | **~0.83** |

> *Results may vary slightly between runs. The best model is automatically selected and saved.*

---

## 🌐 Web Dashboard

The Flask-powered dashboard features:
- **Dark glassmorphism design** with animated backgrounds
- **Real-time fraud prediction** with probability scoring
- **Sample transaction loading** (legitimate & fraud examples)
- **Model performance comparison table**
- **Responsive design** for desktop and mobile

---

## 📊 Notebook Analysis

The Jupyter notebook (`main.ipynb`) includes:
1. **Data Loading & Exploration** — Missing values, duplicates, distribution analysis
2. **Premium Visualizations** — Class distribution, amount analysis, time patterns, correlation heatmaps
3. **Feature Engineering** — Log-transformed amounts, hour extraction, robust scaling
4. **SMOTE Resampling** — Balancing the 577:1 class ratio
5. **Multi-Model Training** — 4 algorithms trained and evaluated
6. **Model Comparison** — Side-by-side ROC & Precision-Recall curves
7. **SHAP Analysis** — Feature importance for the best model
8. **Model Persistence** — Saving artifacts for deployment

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **ML** | scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| **Data** | Pandas, NumPy |

---

## 📄 License

This project is for educational purposes. The dataset is provided by [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
