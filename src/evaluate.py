import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,confusion_matrix,precision_recall_curve,precision_score,recall_score
from data import DataExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

data_loader = DataExtractor()
X_test, y_test = data_loader.test_data()

model_files = {
    "Scaled Class Weights": "model_0.joblib",
    "SMOTE Oversampling": "model_1.joblib",
    "Random Undersampling": "model_2.joblib",
    "Triple Hybrid": "model_3.joblib",
}

metrics_summary = {}

plt.figure(figsize=(9, 7))

for name, filename in model_files.items():
    model_path = MODEL_DIR / filename
    model = joblib.load(model_path)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)

    pr_auc = float(average_precision_score(y_test, y_probs))
    recall = float(recall_score(y_test, y_preds))
    precision = float(precision_score(y_test, y_preds, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_test, y_preds).ravel()

    metrics_summary[name] = {
        "PR_AUC": round(pr_auc, 4),
        "Recall": round(recall, 4),
        "Precision": round(precision, 4),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "True_Negatives": int(tn),
    }

    precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recalls, precisions, label=f"{name} (AUC = {pr_auc:.4f})", lw=2)

baseline_rate = y_test.sum() / len(y_test)
plt.axhline(y=baseline_rate,color="navy",linestyle="--",label=f"No-Skill Baseline ({baseline_rate:.4f})")
plt.xlabel("Recall (Frauds Caught)")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves for different Imbalance Strategies")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(FIGURES_DIR / "pr_curves.png", dpi=300, bbox_inches="tight")
plt.close()

with open(RESULTS_DIR / "metrics.json", "w") as f:
  json.dump(metrics_summary, f, indent=4)

print(f"Metrics saved to {RESULTS_DIR / 'metrics.json'}")
print(f"PR curve plot saved to {FIGURES_DIR / 'pr_curves.png'}")