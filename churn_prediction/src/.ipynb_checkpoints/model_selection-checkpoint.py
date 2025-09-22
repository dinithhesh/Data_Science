"""
Automatic Model Selection with Performance-Based Optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSelector:
    def __init__(self, config_path="configs/model_config.json"):
        self.config = self.load_config(config_path)
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.results = {}
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_data(self, data_path="data/processed/rfm_with_churn.csv"):
        df = pd.read_csv(data_path)
        X = df[['Recency', 'Frequency', 'Monetary']]
        y = df['churn']
        logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def create_models(self):
        """Create models with configured parameters"""
        models = {}
        model_configs = self.config['model_configs']
        
        # Check if hyperparameter tuning is enabled for each model
        if model_configs['RandomForest'].get('hyperparameter_tuning', False):
            models['RandomForest'] = RandomForestClassifier(
                **model_configs['RandomForest']['default_params']
            )
        
        if model_configs['XGBoost'].get('hyperparameter_tuning', False):
            models['XGBoost'] = XGBClassifier(
                **model_configs['XGBoost']['default_params']
            )
        
        if model_configs['LogisticRegression'].get('hyperparameter_tuning', False):
            models['LogisticRegression'] = LogisticRegression(
                **model_configs['LogisticRegression']['default_params']
            )
        
        return models
    
    def hyperparameter_tuning(self, model, model_name, X, y):
        """Perform model-specific hyperparameter tuning"""
        param_grid = self.config['model_configs'][model_name]['param_grid']
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.config['model_selection']['cv_folds'],
            scoring=self.config['model_selection']['primary_metric'],
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        logger.info(f"{model_name} best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, model_name, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.config['model_selection']['cv_folds'],
            scoring=self.config['model_selection']['primary_metric'],
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_auc': roc_auc_score(y_test, y_pred_proba),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'model': model
        }
        
        return metrics
    
    def select_best_model(self):
        """Main method to select best model"""
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['model_selection']['test_size'],
            random_state=self.config['model_selection']['random_state'],
            stratify=y
        )
        
        models = self.create_models()
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"🔍 Evaluating {model_name}...")
            
            # Check if hyperparameter tuning is enabled for the model
            if self.config['model_configs'][model_name].get('hyperparameter_tuning', False):
                tuned_model = self.hyperparameter_tuning(model, model_name, X_train, y_train)
            else:
                tuned_model = model
            
            metrics = self.evaluate_model(tuned_model, model_name, X_train, X_test, y_train, y_test)
            results[model_name] = metrics
            
            logger.info(f"    {model_name}: AUC = {metrics['test_auc']:.4f} ± {metrics['cv_std']:.4f}")
        
        if not results:
            logger.error("No models were configured or evaluated. Please check your model_config.json.")
            return None, None, {}

        # Select the best model based on the primary metric (AUC)
        primary_metric = self.config['model_selection']['primary_metric']
        
        best_model_name = max(results.items(), key=lambda x: x[1]['test_auc'])[0]
        best_score = results[best_model_name]['test_auc']
        best_model = results[best_model_name]['model']
        
        logger.info(f"🏆 BEST MODEL: {best_model_name} (AUC: {best_score:.4f})")
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_score = best_score
        self.results = results
        
        return best_model_name, best_model, results
    
    def save_results(self):
        """Save selection results and metadata"""
        Path("models").mkdir(exist_ok=True)
        
        joblib.dump(self.best_model, "models/Best_Churn_Model.pkl")
        
        results_data = {
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'all_results': {
                name: {k: v for k, v in metrics.items() if k != 'model'}
                for name, metrics in self.results.items()
            },
            'selection_date': datetime.now().isoformat(),
            'model_parameters': self.best_model.get_params()
        }
        
        with open("models/model_selection_results.json", "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"✅ Saved {self.best_model_name} as best model")

if __name__ == '__main__':
    # This block is for standalone testing and will not be called when imported.
    selector = ModelSelector(config_path="configs/model_config.json")
    best_model_name, best_model, results = selector.select_best_model()
    selector.save_results()