import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_and_evaluate():
    # 1. Resolve Project Root Dynamically
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    data_dir = os.path.join(project_root, "data", "processed")
    model_dir = os.path.join(project_root, "models")
    fig_dir = os.path.join(project_root, "docs", "figures")
    doc_dir = os.path.join(project_root, "docs")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(doc_dir, exist_ok=True)

    # 2. Load Processed Datasets
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    print(f" Datasets Loaded: Train shape = {X_train.shape}, Test shape = {X_test.shape}")

    # 3. Define Candidate Models
    models = {
        "Logistic Regression (Baseline)": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000, random_state=42))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.08, random_state=42, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.08, random_state=42, verbose=-1)
    }

    results = []
    best_model = None
    best_f1 = -1.0
    best_model_name = ""

    print("\n⚡ Starting Model Training & Comparison...\n" + "="*50)

    # 4. Train and Evaluate Each Model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        print(f"🔹 {name}: Recall = {rec:.4f} | F1-Score = {f1:.4f} | ROC-AUC = {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("\n" + "="*50)
    print(f" Best Performing Model: {best_model_name} (F1-Score = {best_f1:.4f})")

    # 5. Save the Best Trained Model Artifact
    best_model_path = os.path.join(model_dir, "best_crop_intervention_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f" Model saved to: {best_model_path}")

    # 6. Save Confusion Matrix
    y_best_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_best_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Action', 'Intervention'], yticklabels=['No Action', 'Intervention'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(fig_dir, "best_model_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f" Confusion matrix plot saved to: {cm_path}")

    # 7. Write Model Evaluation Documentation
    report_path = os.path.join(doc_dir, "model_comparison_report.md")
    with open(report_path, "w") as f:
        f.write("# Model Comparison & Performance Evaluation\n\n")
        f.write("## Algorithmic Comparison Table\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n## Champion Model Selection\n")
        f.write(f"- **Selected Champion Model:** `{best_model_name}`\n")
        f.write(f"- **F1-Score:** `{best_f1:.4f}`\n")
        f.write("- **Primary Evaluation Justification:** In agricultural decision support, Recall is prioritized to minimize False Negatives (unnoticed crop risk).\n")

    print(f" Model evaluation report saved to: {report_path}")

if __name__ == "__main__":
    train_and_evaluate()