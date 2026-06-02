"""
Credit Card Fraud Detection — Model Training Script
=====================================================
Trains multiple ML models, compares them, and saves the best one
for use by the Flask web application.

Usage:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("  Credit Card Fraud Detection — Model Training")
    print("=" * 60)

    # ── 1. Load Data ────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv("creditcard.csv")
    print(f"  Loaded {len(df):,} transactions with {df.shape[1]} features")
    fraud_count = (df['Class'] == 1).sum()
    legit_count = (df['Class'] == 0).sum()
    print(f"  Legitimate: {legit_count:,} | Fraud: {fraud_count:,}")

    # ── 2. Preprocess ──────────────────────────────────────────
    print("\n[2/6] Preprocessing data...")
    df['log_Amount'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] / 3600).astype(int) % 24
    df_processed = df.drop(['Amount', 'Time'], axis=1)

    X = df_processed.drop('Class', axis=1)
    y = df_processed['Class']

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 3. SMOTE ───────────────────────────────────────────────
    print("\n[3/6] Applying SMOTE to balance training data...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  Resampled: {len(X_res):,} samples (balanced)")

    # ── 4. Train Models ────────────────────────────────────────
    print("\n[4/6] Training models...")
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=legit_count / fraud_count,
            eval_metric='aucpr', random_state=42,
            n_estimators=100, use_label_encoder=False
        ),
        "LightGBM": LGBMClassifier(
            scale_pos_weight=legit_count / fraud_count,
            random_state=42, n_estimators=100, verbose=-1
        ),
    }

    results = {}
    best_model_name = None
    best_f1 = 0

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc_roc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "auc_roc": round(auc_roc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_score": round(f1, 4),
            "precision_fraud": round(report['1']['precision'], 4),
            "recall_fraud": round(report['1']['recall'], 4),
            "accuracy": round(report['accuracy'], 4),
            "confusion_matrix": cm.tolist(),
        }

        print(f"    AUC-ROC: {auc_roc:.4f} | PR AUC: {pr_auc:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    # ── 5. Compare ─────────────────────────────────────────────
    print("\n[5/6] Model Comparison:")
    print(f"  {'Model':<25} {'AUC-ROC':>8} {'PR AUC':>8} {'F1':>8} {'Recall':>8}")
    print("  " + "-" * 58)
    for name, r in results.items():
        marker = " <-- BEST" if name == best_model_name else ""
        print(f"  {name:<25} {r['auc_roc']:>8.4f} {r['pr_auc']:>8.4f} "
              f"{r['f1_score']:>8.4f} {r['recall_fraud']:>8.4f}{marker}")

    print(f"\n  Best model: {best_model_name} (F1 = {best_f1:.4f})")

    # ── 6. Save ────────────────────────────────────────────────
    print("\n[6/6] Saving artifacts...")
    os.makedirs("models", exist_ok=True)

    # Save best model
    joblib.dump(models[best_model_name], "models/best_model.pkl")
    print(f"  Saved best model -> models/best_model.pkl")

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"  Saved scaler -> models/scaler.pkl")

    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "feature_names": feature_names,
        "results": results,
        "dataset_info": {
            "total_transactions": len(df),
            "fraud_transactions": int(fraud_count),
            "legit_transactions": int(legit_count),
            "fraud_ratio": round(fraud_count / len(df) * 100, 4),
        }
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata -> models/metadata.json")

    print("\n" + "=" * 60)
    print("  Training complete! Run 'python app.py' to start the web UI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
