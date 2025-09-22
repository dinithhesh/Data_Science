"""
Model Deployment Script for Azure ML
Deploys trained customer churn prediction model to Azure ML endpoints
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import ServicePrincipalAuthentication
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDeployer:
    """
    Class to handle model deployment to Azure ML
    """
    
    def __init__(self, config_path: str = "../config/config.json"):
        """
        Initialize model deployer with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.ws = None
        self.model = None
        self.environment = None
        self.service = None
    
    def load_config(self, config_path: str) -> dict:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def connect_to_workspace(self):
        """
        Connect to Azure ML workspace
        """
        try:
            azure_config = self.config['azure_ml']
            
            # Check if we're running in Azure ML (service principal auth)
            if all(key in os.environ for key in ['AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET', 'AZURE_TENANT_ID']):
                auth = ServicePrincipalAuthentication(
                    tenant_id=os.environ['AZURE_TENANT_ID'],
                    service_principal_id=os.environ['AZURE_CLIENT_ID'],
                    service_principal_password=os.environ['AZURE_CLIENT_SECRET']
                )
                logger.info("Using service principal authentication")
            else:
                auth = None
                logger.info("Using default authentication")
            
            # Connect to workspace
            self.ws = Workspace(
                subscription_id=azure_config['subscription_id'],
                resource_group=azure_config['resource_group'],
                workspace_name=azure_config['workspace_name'],
                auth=auth
            )
            
            logger.info(f"Connected to Azure ML workspace: {self.ws.name}")
            logger.info(f"Workspace location: {self.ws.location}")
            
        except Exception as e:
            logger.error(f"Error connecting to workspace: {str(e)}")
            raise
    
    def register_model(self, model_path: str = "../models/best_model.pkl"):
        """
        Register model in Azure ML model registry
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Registered model object
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            deployment_config = self.config['deployment']
            
            # Register model
            self.model = Model.register(
                workspace=self.ws,
                model_path=str(model_path),
                model_name=deployment_config['model_name'],
                tags={
                    "version": deployment_config['model_version'],
                    "framework": "XGBoost",
                    "task": "classification",
                    "description": "Customer churn prediction model"
                },
                description="XGBoost model for customer churn prediction",
                model_framework=Model.Framework.SCIKITLEARN
            )
            
            logger.info(f"Model registered: {self.model.name}")
            logger.info(f"Model ID: {self.model.id}")
            logger.info(f"Model version: {self.model.version}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise
    
    def create_environment(self):
        """
        Create Azure ML environment for deployment
        """
        try:
            env_config = self.config['environment']
            deployment_config = self.config['deployment']
            
            # Create conda dependencies
            conda_dep = CondaDependencies()
            
            # Add conda dependencies
            for pkg in env_config['conda_dependencies']:
                if '=' in pkg:
                    name, version = pkg.split('=')
                    conda_dep.add_conda_package(f"{name}={version}")
                else:
                    conda_dep.add_conda_package(pkg)
            
            # Add pip dependencies
            for pkg in env_config['pip_dependencies']:
                conda_dep.add_pip_package(pkg)
            
            # Create environment
            self.environment = Environment(name=deployment_config['environment_name'])
            self.environment.python.conda_dependencies = conda_dep
            self.environment.docker.enabled = True
            self.environment.docker.base_image = None  # Use Azure ML base image
            
            logger.info(f"Environment created: {self.environment.name}")
            
            return self.environment
            
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            raise
    
    def create_inference_config(self):
        """
        Create inference configuration for deployment
        
        Returns:
            InferenceConfig object
        """
        try:
            # Use the scoring script from the same directory
            scoring_script = "score.py"
            if not os.path.exists(scoring_script):
                scoring_script = "../src/score.py"
            
            inference_config = InferenceConfig(
                entry_script=scoring_script,
                environment=self.environment,
                source_directory=".",
                description="Customer churn prediction inference configuration"
            )
            
            logger.info("Inference configuration created")
            return inference_config
            
        except Exception as e:
            logger.error(f"Error creating inference config: {str(e)}")
            raise
    
    def create_deployment_config(self):
        """
        Create deployment configuration for ACI
        
        Returns:
            AciWebservice deployment config
        """
        try:
            deployment_config = self.config['deployment']
            api_config = self.config['api']
            
            aci_config = AciWebservice.deploy_configuration(
                cpu_cores=deployment_config['cpu_cores'],
                memory_gb=deployment_config['memory_gb'],
                auth_enabled=deployment_config['auth_enabled'],
                description="Customer Churn Prediction Service",
                tags={
                    "project": "customer-segmentation",
                    "framework": "xgboost",
                    "version": deployment_config['model_version']
                },
                enable_app_insights=True,
                ssl_enabled=True,
                ssl_cert_pem_file=None,
                ssl_key_pem_file=None,
                ssl_cname=None,
                location=self.ws.location,
                dns_name_label=f"churn-pred-{datetime.now().strftime('%Y%m%d')}",
                primary_key=None,
                secondary_key=None,
                collect_model_data=True,
                cmk_vault_base_url=None,
                cmk_key_name=None,
                cmk_key_version=None
            )
            
            logger.info("ACI deployment configuration created")
            return aci_config
            
        except Exception as e:
            logger.error(f"Error creating deployment config: {str(e)}")
            raise
    
    def deploy_model(self):
        """
        Deploy model to Azure Container Instances
        """
        try:
            logger.info("Starting model deployment...")
            
            # Create inference config
            inference_config = self.create_inference_config()
            
            # Create deployment config
            deployment_config = self.create_deployment_config()
            
            # Deploy model
            self.service = Model.deploy(
                workspace=self.ws,
                name=self.config['deployment']['model_name'],
                models=[self.model],
                inference_config=inference_config,
                deployment_config=deployment_config,
                overwrite=True
            )
            
            # Wait for deployment to complete
            self.service.wait_for_deployment(show_output=True)
            
            logger.info(f"Model deployed successfully!")
            logger.info(f"Service name: {self.service.name}")
            logger.info(f"Service state: {self.service.state}")
            logger.info(f"Scoring URI: {self.service.scoring_uri}")
            
            if self.service.scoring_uri:
                logger.info("✅ Deployment completed successfully!")
                logger.info(f"Scoring endpoint: {self.service.scoring_uri}")
                
                # Get authentication keys if enabled
                if self.config['deployment']['auth_enabled']:
                    keys = self.service.get_keys()
                    logger.info(f"Primary key: {keys[0]}")
                    logger.info(f"Secondary key: {keys[1] if len(keys) > 1 else 'N/A'}")
            
            return self.service
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise
    
    def test_deployment(self):
        """
        Test the deployed model endpoint
        """
        try:
            if not self.service:
                raise ValueError("Service not deployed. Call deploy_model() first.")
            
            # Test data
            test_data = {
                "features": [30, 5, 60000]  # [Recency, Frequency, Monetary]
            }
            
            # Test the endpoint
            result = self.service.run(input_data=json.dumps(test_data))
            
            logger.info("✅ Deployment test successful!")
            logger.info(f"Test result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing deployment: {str(e)}")
            raise
    
    def update_model(self, new_model_path: str):
        """
        Update existing deployed model
        
        Args:
            new_model_path: Path to the new model file
        """
        try:
            # Register new model version
            new_model = self.register_model(new_model_path)
            
            # Update the service
            self.service.update(models=[new_model])
            self.service.wait_for_deployment(show_output=True)
            
            logger.info(f"Model updated to version {new_model.version}")
            
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            raise
    
    def delete_deployment(self):
        """
        Delete the deployed service
        """
        try:
            if self.service:
                self.service.delete()
                logger.info("Service deleted successfully")
            else:
                logger.warning("No service to delete")
                
        except Exception as e:
            logger.error(f"Error deleting deployment: {str(e)}")
            raise
    
    def get_deployment_status(self):
        """
        Get deployment status
        """
        try:
            if self.service:
                status = self.service.get_state()
                logger.info(f"Deployment status: {status}")
                return status
            else:
                logger.warning("No service deployed")
                return None
                
        except Exception as e:
            logger.error(f"Error getting deployment status: {str(e)}")
            raise

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy model to Azure ML")
    parser.add_argument("--model-path", type=str, default="../models/best_model.pkl", 
                       help="Path to the model file")
    parser.add_argument("--config-path", type=str, default="../config/config.json", 
                       help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Test deployment after deploying")
    parser.add_argument("--delete", action="store_true", help="Delete existing deployment")
    
    args = parser.parse_args()
    
    try:
        # Initialize deployer
        deployer = ModelDeployer(config_path=args.config_path)
        
        # Connect to workspace
        deployer.connect_to_workspace()
        
        if args.delete:
            # Delete existing deployment
            service_name = deployer.config['deployment']['model_name']
            try:
                service = Webservice(deployer.ws, name=service_name)
                service.delete()
                logger.info(f"Deleted service: {service_name}")
            except Exception as e:
                logger.warning(f"Service {service_name} not found or already deleted: {e}")
            return
        
        # Register model
        deployer.register_model(args.model_path)
        
        # Create environment
        deployer.create_environment()
        
        # Deploy model
        service = deployer.deploy_model()
        
        # Test deployment if requested
        if args.test:
            deployer.test_deployment()
        
        # Print deployment information
        print("\n" + "="*60)
        print("🚀 DEPLOYMENT SUCCESSFUL!")
        print("="*60)
        print(f"Service Name: {service.name}")
        print(f"Scoring URI: {service.scoring_uri}")
        print(f"Swagger URI: {service.swagger_uri}")
        print(f"State: {service.state}")
        
        if deployer.config['deployment']['auth_enabled']:
            keys = service.get_keys()
            print(f"Primary Key: {keys[0]}")
        
        print("\n📋 Example usage:")
        print(f"curl -X POST {service.scoring_uri} \\")
        print("  -H \"Content-Type: application/json\" \\")
        print("  -H \"Authorization: Bearer YOUR_PRIMARY_KEY\" \\")
        print("  -d '{\"features\": [30, 5, 60000]}'")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()