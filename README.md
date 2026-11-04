# Credit Card Fraud Detection — Imbalanced Classification Benchmark

An end-to-end machine learning model evaluating cost-sensitive learning against synthetic resampling strategies on severely imbalanced financial transaction data ($0.172\%$ positive fraud ratio).

Rather than relying on default accuracy, repo implements leak-free cross-validation pipelines using `imbalanced-learn` and `XGBoost` optimizing directly for Precision-Recall AUC (`average_precision`) and analyzes False Positive vs. False Negative trade-offs.

---

## Project Structure

```text
imbalanced-classification/
├── datasets/
│   ├── Credit-Card-Data/
│   │   └── creditcard.csv
│   └── README.md
├── models/
│   ├── model_0.joblib
│   ├── model_1.joblib
│   ├── model_2.joblib
│   └── model_3.joblib
├── notebooks/
│   └── 01_eda.ipynb
├── results/
│   ├── figures/
│   │   └── pr_curves.png
│   └── metrics.json
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Engineering Highlights & Methodology

### 1. Zero Data Leakage Architecture

* **Feature Scaling:** `Time` and `Amount` are scaled using `StandardScaler` fit strictly on the 80% training partition prior to transforming the 20% holdout test partition (`V1`–`V28` are already PCA-transformed).
* **In-Fold Resampling:** All oversampling (`SMOTE`) and undersampling (`RandomUnderSampler`) operations are done inside `imblearn.pipeline.Pipeline`. During 5-fold `StratifiedKFold` cross-validation, synthetic points are generated strictly inside each training fold, preventing data leakage into validation splits.

### 2. Evaluated Imbalance Strategies

All four pipelines utilize a tuned `XGBClassifier` (`n_estimators=600`, `learning_rate=0.03`, `max_depth=5`, `max_delta_step=10`, `grow_policy='lossguide'`, `subsample=0.7`, `colsample_bytree=0.6`):

1. **Scaled Class Weights (`model_0`):** Modifies the gradient boosting loss function via `scale_pos_weight = sqrt(neg / pos)` without altering the data distribution.
2. **SMOTE Oversampling (`model_1`):** Synthesizes minority class examples in feature space (`sampling_strategy=0.10`) inside each training fold.
3. **Random Undersampling (`model_2`):** Downsamples the majority class (`sampling_strategy=0.10`) to evaluate decision boundary degradation.
4. **Triple Hybrid (`model_3`):** Combines moderate `SMOTE(0.10)` with `RandomUnderSampler(0.25)` and damped cost-sensitive weighting (`scale_pos_weight=2`) to balance recall and precision.

---

## Benchmark Results

### 5-Fold Stratified Cross-Validation (`X_train`)

| Strategy | Artifact | 5-Fold CV PR-AUC |
| --- | --- | --- |
| **1. Scaled Class Weights** | `model_0.joblib` | **0.8593** |
| **2. SMOTE Oversampling** | `model_1.joblib` | 0.8565 |
| **3. Triple Hybrid** | `model_3.joblib` | 0.8507 |
| **4. Random Undersampling** | `model_2.joblib` | 0.7881 |

### Holdout Test Set Evaluation (`X_test`)

Out-of-sample metrics and Precision-Recall curves are exported to `results/metrics.json` and `results/figures/pr_curves.png` via `src/evaluate.py`.

### Key Engineering Takeaway: Why Algorithmic Weighting Beats SMOTE

In high-dimensional financial datasets, **damped class weighting (`scale_pos_weight = sqrt(neg / pos)`) outperforms SMOTE** in both PR-AUC and precision:

* **Probability Calibration:** Resampling methods (`SMOTE` and `RandomUnderSampler`) artificially inflate the prior probability of fraud during training, shifting predicted probabilities upward and generating excess False Positives at the standard `0.50` decision threshold.
* **Boundary Preservation:** Random undersampling discards ~90% of legitimate transactions, destroying the decision boundary for normal user behavior and causing false alarms to spike.
* **Compute Efficiency:** `scale_pos_weight` requires zero synthetic data generation, trains significantly faster, and minimizes customer friction (false card blocks) while maintaining top-tier fraud recall.
---

## Setup & Usage

### 1. Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download the [ULB Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` inside `datasets/Credit-Card-Data/`.

### 3. Train & Cross-Validate Models

Execute 5-fold Stratified Cross-Validation across all four strategies and serialize the fitted pipelines to `models/`:

```bash
python src/train.py
```

### 4. Evaluate on Holdout Test Set

Run inference on the 20% holdout test split to generate `results/metrics.json` and plot the comparative Precision-Recall curves in `results/figures/pr_curves.png`:

```bash
python src/evaluate.py
```
