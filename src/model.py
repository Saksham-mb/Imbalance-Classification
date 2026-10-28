from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from math import sqrt

def model_weight_scaled(X, y):
    weight_scale=sqrt(((y==0).sum())/((y==1).sum()))
    model_pipeline=Pipeline(steps=[
        ('model', XGBClassifier(random_state=42, 
                                scale_pos_weight=weight_scale))
    ])
    params_grid = {
        'model__gamma': [0],
        'model__reg_lambda': [0],
        'model__reg_alpha': [0],
        'model__max_leaves': [32],
        'model__n_estimators': [2500],
        'model__eta': [0.01]
    }
    gscv=GridSearchCV(model_pipeline, params_grid, scoring='average_precision', n_jobs=-1, 
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                        verbose=2)
    gscv.fit(X, y)
    return gscv.best_estimator_
def model_SMOTE(X, y):
    model_pipeline=Pipeline(steps=[
        ('smote', SMOTE(random_state=42, sampling_strategy=0.10)),
        ('model', XGBClassifier(random_state=42))
    ])
    params_grid = {
        'model__gamma': [0],
        'model__reg_lambda': [0],
        'model__reg_alpha': [0],
        'model__max_leaves': [32],
        'model__n_estimators': [2500],
        'model__eta': [0.01]
    }
    gscv=GridSearchCV(model_pipeline, params_grid, scoring='average_precision', n_jobs=-1, 
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        verbose=2)
    gscv.fit(X, y)
    return gscv.best_estimator_
def model_undersampling(X, y):
    model_pipeline=Pipeline(steps=[
        ('undersampler', RandomUnderSampler(random_state=42, sampling_strategy=0.1)),
        ('model', XGBClassifier(random_state=42))
    ])
    params_grid = {
        'model__gamma': [0],
        'model__reg_lambda': [0],
        'model__reg_alpha': [0],
        'model__max_leaves': [32],
        'model__n_estimators': [2500],
        'model__eta': [0.01]
    }
    gscv=GridSearchCV(model_pipeline, params_grid, scoring='average_precision', n_jobs=-1, 
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        verbose=2)
    gscv.fit(X, y)
    return gscv.best_estimator_
def model_hybrid(X, y):
    model_pipeline=Pipeline(steps=[
        ('undersampler', RandomUnderSampler(random_state=42, sampling_strategy=0.25)),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.10)),
        ('model', XGBClassifier(random_state=42, scale_pos_weight=4))
    ])
    params_grid = {
        'model__gamma': [0],
        'model__reg_lambda': [0],
        'model__reg_alpha': [0],
        'model__max_leaves': [32],
        'model__n_estimators': [2500],
        'model__eta': [0.01]
    }
    gscv=GridSearchCV(model_pipeline, params_grid, scoring='average_precision', n_jobs=-1, 
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        verbose=2)
    gscv.fit(X, y)
    return gscv.best_estimator_