"""
Model-Specific Deployment Configuration
Handles deployment settings based on the best performing model type
"""

import json
import logging
from pathlib import Path
from azureml.core.webservice import AciWebservice, AksWebservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Model
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self, config_path="configs/deployment_config.json"):
        self.config = self.load_config(config_path)
        self.model = None
        self.model_name = None
        self.model_type = None
        
    def load_config(self, config_path):
        """Load deployment configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Deployment config not found, using defaults")
            return {
                "default_deployment": {
                    "cpu_cores": 1,
                    "memory_gb": 2,
                    "auth_enabled": True
                }
            }
    
    def load_model_info(self):
        """Load model information from selection results"""
        try:
            with open("models/model_selection_results.json", "r") as f:
                results = json.load(f)
            
            self.model_name = results['best_model']
            self.model_type = results['best_model']
            self.model = joblib.load("models/Best_Churn_Model.pkl")
            
            logger.info(f"Loaded model: {self.model_name} ({self.model_type})")
            
        except FileNotFoundError:
            logger.error("Model selection results not found. Run model selection first.")
            raise
    
    def get_model_specific_config(self):
        """Get deployment configuration based on model type"""
        model_config = self.config.get('model_specific_configs', {}).get(
            self.model_type, 
            self.config['default_deployment']
        )
        
        # Adjust based on model type characteristics
        if self.model_type == "RandomForest":
            # RF can be memory intensive with many trees
            model_config['memory_gb'] = max(model_config.get('memory_gb', 2), 4)
            model_config['cpu_cores'] = max(model_config.get('cpu_cores', 1), 2)
            model_config['description'] = "Random Forest Churn Prediction Model"
            model_config['tags'] = {
                "model_type": "RandomForest",
                "n_estimators": str(getattr(self.model, 'n_estimators', 'unknown')),
                "complexity": "high"
            }
            
        elif self.model_type == "XGBoost":
            # XGBoost is efficient but benefits from more CPU
            model_config['cpu_cores'] = max(model_config.get('cpu_cores', 1), 2)
            model_config['description'] = "XGBoost Churn Prediction Model"
            model_config['tags'] = {
                "model_type": "XGBoost",
                "complexity": "medium"
            }
            
        elif self.model_type == "LogisticRegression":
            # LR is lightweight
            model_config['description'] = "Logistic Regression Churn Model"
            model_config['tags'] = {
                "model_type": "LogisticRegression",
                "complexity": "low"
            }
        
        return model_config
    
    def create_deployment_config(self, compute_target="aci"):
        """Create deployment configuration for the selected model"""
        model_config = self.get_model_specific_config()
        
        if compute_target.lower() == "aci":
            deployment_config = AciWebservice.deploy_configuration(
                cpu_cores=model_config['cpu_cores'],
                memory_gb=model_config['memory_gb'],
                auth_enabled=model_config['auth_enabled'],
                description=model_config.get('description', 'Churn Prediction Model'),
                tags=model_config.get('tags', {}),
                enable_app_insights=model_config.get('enable_app_insights', True),
                ssl_enabled=model_config.get('ssl_enabled', True),
                dns_name_label=f"churn-{self.model_type.lower()}-{hash(self.model_type)}"[:32]
            )
            
        elif compute_target.lower() == "aks":
            deployment_config = AksWebservice.deploy_configuration(
                cpu_cores=model_config['cpu_cores'],
                memory_gb=model_config['memory_gb'],
                auth_enabled=model_config['auth_enabled'],
                description=model_config.get('description', 'Churn Prediction Model'),
                tags=model_config.get('tags', {}),
                enable_app_insights=model_config.get('enable_app_insights', True)
            )
        
        logger.info(f"Created {compute_target.upper()} deployment config for {self.model_type}")
        return deployment_config
    
    def create_inference_config(self):
        """Create inference configuration with model-specific settings"""
        # Use the appropriate scoring script
        scoring_script = "score.py"
        
        # Model-specific environment settings
        environment = Environment.from_conda_specification(
            name=f"churn-{self.model_type}-env",
            file_path="environment.yml"
        )
        
        inference_config = InferenceConfig(
            entry_script=scoring_script,
            environment=environment,
            source_directory=".",
            description=f"Inference config for {self.model_type} model"
        )
        
        return inference_config
    
    def deploy_model(self, workspace, compute_target="aci"):
        """Deploy the selected model"""
        try:
            self.load_model_info()
            
            # Register model
            model = Model.register(
                workspace=workspace,
                model_path="models/Best_Churn_Model.pkl",
                model_name=f"churn-{self.model_type}-model",
                tags={
                    "model_type": self.model_type,
                    "deployment_target": compute_target,
                    "version": "1.0.0"
                },
                description=f"Best performing churn prediction model ({self.model_type})"
            )
            
            # Create configurations
            inference_config = self.create_inference_config()
            deployment_config = self.create_deployment_config(compute_target)
            
            # Deploy service
            service_name = f"churn-{self.model_type}-service"
            
            service = Model.deploy(
                workspace=workspace,
                name=service_name,
                models=[model],
                inference_config=inference_config,
                deployment_config=deployment_config,
                overwrite=True
            )
            
            service.wait_for_deployment(show_output=True)
            
            logger.info(f"✅ {self.model_type} model deployed successfully!")
            logger.info(f"Service URL: {service.scoring_uri}")
            logger.info(f"Swagger URL: {service.swagger_uri}")
            
            return service
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            raise
    
    def test_deployment(self, service):
        """Test the deployed model"""
        test_data = {
            "features": [30, 5, 50000]  # [Recency, Frequency, Monetary]
        }
        
        try:
            result = service.run(input_data=json.dumps(test_data))
            logger.info(f"✅ Deployment test successful: {result}")
            return result
        except Exception as e:
            logger.error(f"❌ Deployment test failed: {e}")
            return None

def main():
    """Example usage"""
    from azureml.core import Workspace
    
    # Connect to workspace
    ws = Workspace.from_config()
    
    # Deploy model
    deployer = ModelDeployer()
    service = deployer.deploy_model(ws, compute_target="aci")
    
    # Test deployment
    if service:
        deployer.test_deployment(service)

if __name__ == "__main__":
    main()