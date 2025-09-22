"""
Azure ML Pipeline for Customer Segmentation and Churn Prediction
End-to-end pipeline for data processing, model training, and deployment
"""

import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from azureml.core import Workspace, Experiment, Dataset, Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep
from azureml.core import ScriptRunConfig
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model
from azureml.core.conda_dependencies import CondaDependencies

# This import is correct and will not cause an error if the class and file names are different.
# Your model_selection.py contains the ModelSelector class.
from src.model_selection import ModelSelector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# This class should be in a separate script (e.g., smart_pipeline.py) if you
# want to use it as a standalone runner. If you want to use it within your
# main pipeline, you'll need to refactor it into a pipeline step.
# For now, I've kept it here as it was in your previous code,
# but note that the ImportError is likely from trying to run it
# from a file named model_selector.py.
class SmartPipeline:
    def __init__(self):
        # Use the ModelSelector class here
        self.selector = ModelSelector(config_path="configs/model_config.json")
        self.best_model_name = None
        self.best_model = None

    def run_pipeline(self):
        """Complete pipeline with automatic model selection"""
        logger.info("🚀 Starting Smart Pipeline...")
        
        # 1. Select best model
        self.best_model_name, self.best_model, results = self.selector.select_best_model()
        
        # 2. Save results
        self.selector.save_results()
        
        logger.info("✅ Pipeline completed successfully!")

class CustomerSegmentationPipeline:
    """
    Main pipeline class for customer segmentation and churn prediction
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the pipeline with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.ws = None
        self.compute_target = None
        self.pipeline = None
        self.experiment = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
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
        # ... (rest of the method is unchanged)

    def get_or_create_compute_target(self):
        """
        Get or create compute target for pipeline
        """
        # ... (rest of the method is unchanged)

    def create_environment(self):
        """
        Create Azure ML environment for pipeline steps
        """
        # ... (rest of the method is unchanged)

    def create_data_processing_step(self, environment):
        """
        Create data processing pipeline step
        """
        # ... (rest of the method is unchanged)

    def create_clustering_step(self, environment, input_data):
        """
        Create customer clustering pipeline step
        """
        # ... (rest of the method is unchanged)

    def create_model_training_step(self, environment, input_data):
        """
        Create model training pipeline step
        """
        # ... (rest of the method is unchanged)

    def create_evaluation_step(self, environment, input_data, models_data):
        """
        Create model evaluation pipeline step
        """
        # ... (rest of the method is unchanged)

    def create_register_model_step(self, environment, models_data, evaluation_data):
        """
        Create model registration pipeline step
        """
        # ... (rest of the method is unchanged)

    def build_pipeline(self):
        """
        Build the complete pipeline
        """
        # ... (rest of the method is unchanged)

    def run_pipeline(self):
        """
        Run the pipeline
        """
        # ... (rest of the method is unchanged)

    def display_pipeline_metrics(self, pipeline_run):
        """
        Display metrics from pipeline run
        """
        # ... (rest of the method is unchanged)

    def schedule_pipeline(self, schedule_name: str, schedule_expression: str = "0 0 * * 0"):
        """
        Schedule pipeline to run periodically
        """
        # ... (rest of the method is unchanged)

    def cleanup(self):
        """
        Clean up resources
        """
        # ... (rest of the method is unchanged)

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="Run Customer Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="config.json", 
                        help="Path to configuration file")
    parser.add_argument("--schedule", action="store_true", 
                        help="Schedule the pipeline for weekly runs")
    parser.add_argument("--run-only", action="store_true", 
                        help="Run pipeline without building (use existing)")
    parser.add_argument("--cleanup", action="store_true", 
                        help="Clean up resources after run")
    parser.add_argument("--smart", action="store_true",
                        help="Run the smart pipeline with automatic model selection")

    args = parser.parse_args()
    
    pipeline = None
    try:
        if args.smart:
            # Run SmartPipeline
            pipeline = SmartPipeline()
            pipeline.run_pipeline()
        else:
            # Run CustomerSegmentationPipeline
            pipeline = CustomerSegmentationPipeline(config_path=args.config)
            
            pipeline.connect_to_workspace()
            pipeline.get_or_create_compute_target()
            
            if not args.run_only:
                pipeline.build_pipeline()
            
            published_pipeline = pipeline.run_pipeline()
            
            if published_pipeline and args.schedule:
                pipeline.schedule_pipeline("weekly-customer-segmentation")
            
            if args.cleanup:
                pipeline.cleanup()
            
        logger.info("🎉 Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise
        
    finally:
        if pipeline and args.cleanup:
            pipeline.cleanup()

if __name__ == "__main__":
    main()