
#Random Forest Specific Optimizations


import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def optimize_random_forest(model_path, config):
    model = joblib.load(model_path)
    
    # Reduce trees for production if needed
    if model.n_estimators > config['optimization']['max_trees_production']:
        optimized_model = RandomForestClassifier(
            n_estimators=config['optimization']['max_trees_production'],
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            min_samples_leaf=model.min_samples_leaf,
            random_state=model.random_state,
            n_jobs=-1
        )
        return optimized_model
    
    return model

def get_feature_importance(model):
    return pd.DataFrame({
        'feature': ['Recency', 'Frequency', 'Monetary'],
        'importance': model.feature_importances_,
        'model_type': 'RandomForest'
    })