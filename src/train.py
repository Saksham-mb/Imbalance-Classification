from model import Models
from data import DataExtractor
import joblib
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent
MODEL_DIR=PROJECT_ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

data_loader=DataExtractor()
X_train, y_train=data_loader.train_data()

cls_models=Models()


model_0, score_0 = cls_models.model_weight_scaled(X_train, y_train)
print(f"Average PR-AUC score across 5 fold cross validation X_train for model with scaled class weights is {score_0:.4f}")
joblib.dump(model_0, MODEL_DIR / 'model_0.joblib')
print("Saved model_0")

model_1, score_1 = cls_models.model_SMOTE(X_train, y_train)
print(f"Average PR-AUC score across 5 fold cross validation X_train for model with SMOTE oversampling is {score_1:.4f}")
joblib.dump(model_1, MODEL_DIR / 'model_1.joblib')
print("Saved model_1")

model_2, score_2 = cls_models.model_undersampling(X_train, y_train)
print(f"Average PR-AUC score across 5 fold cross validation X_train for model with Random Undersampling is {score_2:.4f}")
joblib.dump(model_2, MODEL_DIR / 'model_2.joblib')
print("Saved model_2")

model_3, score_3 = cls_models.model_hybrid(X_train, y_train)
print(f"Average PR-AUC score across 5 fold cross validation X_train for model with a hybrid approach is {score_3:.4f}")
joblib.dump(model_3, MODEL_DIR / 'model_3.joblib')
print("Saved model_3")