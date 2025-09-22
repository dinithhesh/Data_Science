#Model Training Module for Customer Churn Prediction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, accuracy_score, f1_score,
                            precision_score, recall_score, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Class to handle model training, evaluation, and optimization
class ModelTrainer:
    
    
    #Initialize the model trainer
    def __init__(self, random_state=42):
        
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_importances = {}
     
     #Prepare data for training   
    def prepare_data(self, X, y, test_size=0.2, scale_features=True):
        
        try:
            logger.info("Preparing data for training...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            logger.info(f"Train target distribution:\n{y_train.value_counts()}")
            logger.info(f"Test target distribution:\n{y_test.value_counts()}")
            
            # Scale features if requested
            if scale_features:
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
                logger.info("Features scaled using StandardScaler")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    #Train multiple models
    def train_models(self, X_train, y_train):
        
        try:
            logger.info("Training multiple models...")
            
            # Define models 
            models = {
                'logistic_regression': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'random_forest': RandomForestClassifier(
                    random_state=self.random_state,
                    n_estimators=100,
                    class_weight='balanced'
                ),
                'xgboost': XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
            }
            
            # Train each model
            for name, model in models.items():
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                self.models[name] = model
                logger.info(f"{name} training completed")
            
            return self.models
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    # Evaluate all trained models
    def evaluate_models(self, X_test, y_test):
        
        try:
            logger.info("Evaluating models...")
            
            results = {}
            
            for name, model in self.models.items():
                logger.info(f"Evaluating {name}...")
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                results[name] = metrics
                
                logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            raise
    
    # Perform hyperparameter tuning for a specific model
    def hyperparameter_tuning(self, X_train, y_train, model_name='xgboost', cv=5):
        
        try:
            logger.info(f"Performing hyperparameter tuning for {model_name}...")
            
            param_grids = {
                'logistic_regression': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
            
            if model_name not in param_grids:
                raise ValueError(f"Hyperparameter grid not defined for {model_name}")
            
            # Get base model
            base_model = self.models.get(model_name)
            if base_model is None:
                # Create new instance if not trained yet
                if model_name == 'logistic_regression':
                    base_model = LogisticRegression(random_state=self.random_state)
                elif model_name == 'random_forest':
                    base_model = RandomForestClassifier(random_state=self.random_state)
                elif model_name == 'xgboost':
                    base_model = XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grids[model_name],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Update the model with best parameters
            self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
            self.best_model = grid_search.best_estimator_
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise
    
    #Get feature importances from model
    def get_feature_importances(self, model_name='xgboost_tuned'):
        
        try:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                logger.warning(f"Model {model_name} doesn't have feature importances")
                return None
            
            # Create feature importance dictionary
            feature_importance_dict = dict(zip(range(len(importances)), importances))
            self.feature_importances[model_name] = feature_importance_dict
            
            return  feature_importances_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importances: {str(e)}")
            raise
    
    #Plot ROC curve for all models
    def plot_roc_curve(self, X_test, y_test, output_path=None):
       
        try:
            plt.figure(figsize=(10, 8))
            
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves - Model Comparison')
            plt.legend(loc='lower right')
            plt.grid(True)
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                logger.info(f"ROC curve saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            raise
    
    #Save trained models to disk
    def save_models(self, output_dir="models"):
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for name, model in self.models.items():
                filename = f"{name}_{timestamp}.pkl"
                filepath = os.path.join(output_dir, filename)
                joblib.dump(model, filepath)
                logger.info(f"Model {name} saved to: {filepath}")
            
            # Save scaler
            scaler_path = os.path.join(output_dir, f"scaler_{timestamp}.pkl")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
            
            # Save best model separately
            if self.best_model:
                best_model_path = os.path.join(output_dir, "best_model.pkl")
                joblib.dump(self.best_model, best_model_path)
                logger.info(f"Best model saved to: {best_model_path}")
            
            # Save results
            results_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to: {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    #Load a trained model from disk
    def load_model(self, model_path):
        
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

# Example usage
def main():
    
    
    # Create sample data for testing
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=3, n_informative=3, n_redundant=0,
        random_state=42, weights=[0.8, 0.2]
    )
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=0.2)
    
    # Train models
    trainer.train_models(X_train, y_train)
    
    # Evaluate models
    results = trainer.evaluate_models(X_test, y_test)
    
    # Hyperparameter tuning
    best_model = trainer.hyperparameter_tuning(X_train, y_train, 'xgboost')
    
    # Get feature importances
    importances = trainer.get_feature_importances('xgboost_tuned')
    print("Feature importances:", importances)
    
    # Plot ROC curve
    trainer.plot_roc_curve(X_test, y_test, "roc_curve.png")
    
    # Save models
    trainer.save_models()
    
    return trainer

if __name__ == "__main__":
    trainer = main()
    print("Model training completed successfully!")