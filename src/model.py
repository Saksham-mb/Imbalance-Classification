from math import sqrt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

class Models:
    def __init__(self):
        self.params = {
        "subsample": 0.7,
        "reg_lambda": 1.0,
        "reg_alpha": 1.0,
        "n_estimators": 600,
        "min_child_weight": 3,
        "max_depth": 5,
        "max_delta_step": 10,
        "learning_rate": 0.03,
        "grow_policy": "lossguide",
        "gamma": 0.5,
        "colsample_bytree": 0.6,
        "random_state": 42,
        "n_jobs": -1,
    }
    def build_and_evaluate(self, pipeline, X, y):
        scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="average_precision", n_jobs=1)
        cv_pr_auc = scores.mean()

        pipeline.fit(X, y)
        return pipeline, cv_pr_auc
    def model_weight_scaled(self, X, y):
        weight_scale = sqrt(((y == 0).sum()) / ((y == 1).sum()))
        pipeline = Pipeline(steps=[
                ("model",XGBClassifier(scale_pos_weight=weight_scale, **self.params))
            ])
        return self.build_and_evaluate(pipeline, X, y)

    def model_SMOTE(self, X, y):
        pipeline = Pipeline(steps=[
                ("smote", SMOTE(random_state=42, sampling_strategy=0.10)),
                ("model", XGBClassifier(**self.params)),
            ]
        )
        return self.build_and_evaluate(pipeline, X, y)

    def model_undersampling(self, X, y):
        pipeline = Pipeline(steps=[
                ("undersampler",RandomUnderSampler(random_state=42, sampling_strategy=0.1)),
                ("model", XGBClassifier(**self.params))
            ])
        return self.build_and_evaluate(pipeline, X, y)

    def model_hybrid(self, X, y):
        pipeline = Pipeline(steps=[
                ("smote", SMOTE(random_state=42, sampling_strategy=0.10)),
                ("undersampler",RandomUnderSampler(random_state=42, sampling_strategy=0.25)),
                ("model", XGBClassifier(scale_pos_weight=2, **self.params))
            ])
        return self.build_and_evaluate(pipeline, X, y)