
#Azure ML Model Deployment Package



import json
import numpy as np
import pandas as pd
import joblib
from azureml.core.model import Model
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
model_features = ['Recency', 'Frequency', 'Monetary']

def init():
    """
    This function is called when the container is initialized/started.
    Typically used to load the model and any other required assets.
    """
    global model
    
    try:
        # Log environment information
        logger.info("Initializing model...")
        logger.info(f"Python version: {os.sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir('.')}")
        
        
        # Try different possible model paths
        possible_model_paths = [
            Model.get_model_path('best_churn_model'),  
            './models/Best_Churn_Model.pkl',           
            '/var/azureml-app/azureml-models/best_churn_model/1/Best_Churn_Model.pkl',
            './Best_Churn_Model.pkl',
            'Best_Churn_Model.pkl'
        ]
        
        model_loaded = False
        
        for model_path in possible_model_paths:
            try:
                if os.path.exists(model_path):
                    logger.info(f"Loading model from: {model_path}")
                    model = joblib.load(model_path)
                    model_loaded = True
                    logger.info("Model loaded successfully")
                    break
                else:
                    logger.info(f"Model path not found: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load from {model_path}: {str(e)}")
                continue
        
        if not model_loaded:
            
            logger.warning("No model file found. Creating mock model for testing.")
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            # Create a simple mock model
            X, y = make_classification(n_samples=100, n_features=3, random_state=42)
            mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
            mock_model.fit(X, y)
            model = mock_model
            logger.info("Mock model created for testing")
            
        # Log model information
        if hasattr(model, 'feature_importances_'):
            logger.info(f"Model feature importances: {model.feature_importances_}")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model features expected: {model_features}")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise

def validate_input_data(input_data):
    """
    Validate the input data format and values
    """
    try:
        # Check if features key exists
        if 'features' not in input_data:
            raise ValueError("Input data must contain 'features' key")
        
        features = input_data['features']
        
        # Check if features is a list
        if not isinstance(features, list):
            raise ValueError("Features must be a list")
        
        # Check if we have exactly 3 features (Recency, Frequency, Monetary)
        if len(features) != 3:
            raise ValueError(f"Expected 3 features, got {len(features)}")
        
        # Convert to numpy array and validate numeric values
        features_array = np.array(features, dtype=float)
        
        # Validate feature ranges (adjust based on your data)
        if features_array[0] < 0:  # Recency should be positive
            raise ValueError("Recency must be positive")
        if features_array[1] < 0:  # Frequency should be positive
            raise ValueError("Frequency must be positive")
        if features_array[2] < 0:  # Monetary should be positive
            raise ValueError("Monetary value must be positive")
            
        return features_array
        
    except ValueError as ve:
        logger.error(f"Input validation error: {ve}")
        raise ve
    except Exception as e:
        logger.error(f"Unexpected error during input validation: {str(e)}")
        raise

def run(raw_data):
    """
    This function is called for every invocation of the endpoint.
    It handles the prediction request.
    
    Parameters:
    raw_data (str): JSON string containing the input data
    
    Returns:
    dict: Prediction results with probabilities
    """
    try:
        logger.info("Received prediction request")
        logger.info(f"Raw data received: {raw_data[:200]}...")  
        
        # Parse the input JSON data
        try:
            data = json.loads(raw_data)
            logger.info(f"Parsed JSON data: {data}")
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            logger.error(error_msg)
            return {
                'error': error_msg,
                'prediction': None,
                'churn_probability': None,
                'no_churn_probability': None,
                'prediction_label': 'Error - Invalid JSON'
            }
        
        # Validate input data
        try:
            features_array = validate_input_data(data)
        except ValueError as ve:
            return {
                'error': str(ve),
                'prediction': None,
                'churn_probability': None,
                'no_churn_probability': None,
                'prediction_label': 'Error - Invalid Input'
            }
        
        # Check if model is loaded
        if model is None:
            error_msg = "Model not loaded. Please check initialization."
            logger.error(error_msg)
            return {
                'error': error_msg,
                'prediction': None,
                'churn_probability': None,
                'no_churn_probability': None,
                'prediction_label': 'Error - Model Not Loaded'
            }
        
        # Reshape for prediction (single sample)
        input_data = features_array.reshape(1, -1)
        logger.info(f"Input data shape: {input_data.shape}")
        logger.info(f"Input values - Recency: {features_array[0]}, Frequency: {features_array[1]}, Monetary: {features_array[2]}")
        
        # Make prediction
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            logger.info(f"Raw prediction: {prediction}")
            logger.info(f"Prediction probabilities: {prediction_proba}")
            
        except Exception as e:
            error_msg = f"Error during model prediction: {str(e)}"
            logger.error(error_msg)
            return {
                'error': error_msg,
                'prediction': None,
                'churn_probability': None,
                'no_churn_probability': None,
                'prediction_label': 'Error - Prediction Failed'
            }
        
        # Get prediction probabilities
        churn_probability = float(prediction_proba[0][1])
        no_churn_probability = float(prediction_proba[0][0])
        
        # Determine prediction label
        prediction_label = 'Churn' if prediction[0] == 1 else 'No Churn'
        
        # Prepare comprehensive response
        result = {
            'prediction': int(prediction[0]),
            'churn_probability': churn_probability,
            'no_churn_probability': no_churn_probability,
            'prediction_label': prediction_label,
            'features_used': model_features,
            'feature_values': {
                'Recency': float(features_array[0]),
                'Frequency': float(features_array[1]),
                'Monetary': float(features_array[2])
            },
            'confidence': max(churn_probability, no_churn_probability),
            'model_type': type(model).__name__,
            'success': True
        }
        
        logger.info(f"Prediction successful: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error during prediction: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'error': error_msg,
            'prediction': None,
            'churn_probability': None,
            'no_churn_probability': None,
            'prediction_label': 'Error - Unexpected',
            'success': False
        }

# For local testing and development
if __name__ == "__main__":
    print("Testing model initialization and prediction...")
    
    # Initialize the model
    try:
        init()
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        exit(1)
    
    # Test cases
    test_cases = [
        {"features": [10, 5, 200]},    
        {"features": [120, 1, 50]},    
        {"features": [60, 3, 150]},   
        {"features": [30, 10, 500]},  
        {"features": [150, 2, 75]}     
    ]
    
    print("\n" + "="*50)
    print("TESTING PREDICTIONS")
    print("="*50)
    
    for i, test_data in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {test_data}")
        
        # Convert to JSON string
        raw_test_data = json.dumps(test_data)
        
        # Make prediction
        result = run(raw_test_data)
        
        if result['success']:
            print(f"✓ Prediction: {result['prediction_label']}")
            print(f"  Churn Probability: {result['churn_probability']:.3f}")
            print(f"  No Churn Probability: {result['no_churn_probability']:.3f}")
            print(f"  Confidence: {result['confidence']:.3f}")
        else:
            print(f"✗ Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "="*50)
    print("TESTING COMPLETED")
    print("="*50)