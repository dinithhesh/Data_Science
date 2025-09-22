
#Logistic Regression Specific Optimizations


import joblib
import pandas as pd
import numpy as np

def optimize_logistic_regression(model_path, config):
    model = joblib.load(model_path)
    return model  # LR doesn't need much optimization

def get_feature_importance(model):
    return pd.DataFrame({
        'feature': ['Recency', 'Frequency', 'Monetary'],
        'importance': np.abs(model.coef_[0]),
        'model_type': 'LogisticRegression'
    })