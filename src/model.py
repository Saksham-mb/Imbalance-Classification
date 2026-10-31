from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from math import sqrt

class models:
    def __init__(self):
        self.params_grid = {
                    'max_delta_step': [1, 5, 10],
                    'min_child_weight': [1, 3, 5],
                    
                    'learning_rate': [0.01, 0.03, 0.05],
                    'n_estimators': [300, 600, 1000],
                    'max_depth': [3, 4, 6],
                    
                    'subsample': [0.7, 0.85, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    
                    'reg_lambda': [1.0, 5.0, 15.0],
                    'reg_alpha': [0.0, 0.5, 2.0],
                    'gamma': [0.0, 0.5, 2.0]
                }
        
    def model_weight_scaled(self, X, y):
        weight_scale=sqrt(((y==0).sum())/((y==1).sum()))
        model_pipeline=Pipeline(steps=[
            ('model', XGBClassifier(random_state=42, 
                                    scale_pos_weight=weight_scale))
        ])
        gscv=GridSearchCV(model_pipeline, self.params_grid, scoring='average_precision', n_jobs=-1, 
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                            verbose=2)
        gscv.fit(X, y)
        return gscv.best_estimator_, gscv.best_score_
    
    def model_SMOTE(self, X, y):
        model_pipeline=Pipeline(steps=[
            ('smote', SMOTE(random_state=42, sampling_strategy=0.10)),
            ('model', XGBClassifier(random_state=42))
        ])
        gscv=GridSearchCV(model_pipeline, self.params_grid, scoring='average_precision', n_jobs=-1, 
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            verbose=2)
        gscv.fit(X, y)
        return gscv.best_estimator_, gscv.best_score_
    
    def model_undersampling(self, X, y):
        model_pipeline=Pipeline(steps=[
            ('undersampler', RandomUnderSampler(random_state=42, sampling_strategy=0.1)),
            ('model', XGBClassifier(random_state=42))
        ])
        gscv=GridSearchCV(model_pipeline, self.params_grid, scoring='average_precision', n_jobs=-1, 
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            verbose=2)
        gscv.fit(X, y)
        return gscv.best_estimator_, gscv.best_score_
    
    def model_hybrid(self, X, y):
        model_pipeline=Pipeline(steps=[
            ('smote', SMOTE(random_state=42, sampling_strategy=0.10)),
            ('undersampler', RandomUnderSampler(random_state=42, sampling_strategy=0.25)),
            ('model', XGBClassifier(random_state=42, scale_pos_weight=4))
        ])
        gscv=GridSearchCV(model_pipeline, self.params_grid, scoring='average_precision', n_jobs=-1, 
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                            verbose=2)
        gscv.fit(X, y)
        return gscv.best_estimator_, gscv.best_score_