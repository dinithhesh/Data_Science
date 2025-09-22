
#XGBoost Specific Optimizations


import joblib
import pandas as pd

def optimize_xgboost(model_path, config):
    model = joblib.load(model_path)
    
    # XGBoost specific optimizations
    if hasattr(model, 'best_iteration') and model.best_iteration:
        # Use early stopping result
        model.n_estimators = model.best_iteration
    
    return model

def get_feature_importance(model):
    return pd.DataFrame({
        'feature': ['Recency', 'Frequency', 'Monetary'],
        'importance': model.feature_importances_,
        'model_type': 'XGBoost'
    })