
#Azure ML Pipeline for Customer Segmentation and Churn Prediction


import argparse
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Azure ML SDK Imports
from azureml.core import Workspace, Experiment, Dataset, Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.core import Pipeline, PipelineData, PublishedPipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Mocks the data processing step
def data_processing(input_dataset: Dataset, processed_data: PipelineData):
    
    logger.info("Running mock data processing step...")
    
    # Generate mock data
    df = pd.DataFrame({
        'CustomerID': range(100),
        'Recency': np.random.randint(1, 365, 100),
        'Frequency': np.random.randint(1, 50, 100),
        'Monetary': np.random.randint(100, 5000, 100),
        'has_churned': np.random.randint(0, 2, 100)
    })
    
    # Save the processed data to the output path
    output_path = processed_data.path_on_datastore
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, "rfm_with_churn.csv"), index=False)
    
    logger.info("Mock data processing completed.")

# Mocks the clustering step
def clustering(input_data: PipelineData, clustered_data: PipelineData, models_dir: PipelineData):
   
    logger.info("Running mock clustering step...")
    
    # Load input data
    input_path = input_data.path_on_datastore
    input_file = os.path.join(input_path, "rfm_with_churn.csv")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded data from: {input_file}")
    
   
    
    # Mock clustering and add a 'segment' column
    df['segment'] = np.random.randint(0, 4, len(df))
    logger.info("Mock clustering completed and 'segment' column added.")

    # Save the clustered data
    output_path = clustered_data.path_on_datastore
    os.makedirs(output_path, exist_ok=True)
    df.to_csv(os.path.join(output_path, "clustered_data.csv"), index=False)
    logger.info(f"Clustered data saved to: {os.path.join(output_path, 'clustered_data.csv')}")
    
    # Mock saving scaler and KMeans model
    models_path = models_dir.path_on_datastore
    os.makedirs(models_path, exist_ok=True)
    Path(os.path.join(models_path, "scaler.pkl")).touch()
    Path(os.path.join(models_path, "kmeans_model.pkl")).touch()
    
    logger.info("Mock clustering completed.")

# Mocks the model training step
def model_training(input_data: PipelineData, models_dir: PipelineData):
   
    logger.info("Running mock model training step...")
    
    from sklearn.dummy import DummyClassifier
    import joblib
    
    # Load input data
    input_path = input_data.path_on_datastore
    df = pd.read_csv(os.path.join(input_path, "clustered_data.csv"))
    
    # Mock training a simple model
    features = ['Recency', 'Frequency', 'Monetary', 'segment']
    X = df[features]
    y = df['has_churned']
    
    dummy_model = DummyClassifier(strategy='prior')
    dummy_model.fit(X, y)
    
    # Save the model
    models_path = models_dir.path_on_datastore
    os.makedirs(models_path, exist_ok=True)
    joblib.dump(dummy_model, os.path.join(models_path, 'Best_Churn_Model.pkl'))
    
    logger.info("Mock model training completed.")

# Mocks the model evaluation step
def model_evaluation(models_dir: PipelineData, evaluation_output: PipelineData):
    
    logger.info("Running mock model evaluation step...")
    
    # Mock evaluation report
    report = {
        "accuracy": 0.85,
        "roc_auc": 0.90,
        "f1_score": 0.88,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save the report
    output_path = evaluation_output.path_on_datastore
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info("Mock evaluation completed.")

# Mocks the model registration step
def register_model(evaluation_data: PipelineData, models_dir: PipelineData):
    
    logger.info("Running mock model registration step...")
    
    # Load evaluation report
    eval_path = evaluation_data.path_on_datastore
    with open(os.path.join(eval_path, "evaluation_report.json"), "r") as f:
        report = json.load(f)
        
    # Mock model registration
    model_name = "customer-churn-model"
    model_path = models_dir.path_on_datastore
    
    logger.info(f"Mock registering model '{model_name}' from path: {model_path}")
    logger.info("Mock model registration completed.")


class CustomerSegmentationPipeline:
    
    #Initialize the pipeline with configuration
    def __init__(self, config_path: str = "config.json"):
        
        
        self.config = self.load_config(config_path)
        self.ws = None
        self.compute_target = None
        self.pipeline = None
        self.experiment = None
    
    # Load configuration from JSON file
    def load_config(self, config_path: str) -> Dict[str, Any]:
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    #Connect to Azure ML workspace
    def connect_to_workspace(self):
        
        logger.info("Connecting to Azure ML Workspace...")
        try:
            auth = ServicePrincipalAuthentication(
                tenant_id=os.getenv("AZURE_TENANT_ID"),
                service_principal_id=os.getenv("AZURE_CLIENT_ID"),
                service_principal_password=os.getenv("AZURE_CLIENT_SECRET")
            )
            self.ws = Workspace.get(
                name=self.config['workspace_name'],
                subscription_id=self.config['subscription_id'],
                resource_group=self.config['resource_group'],
                auth=auth
            )
            logger.info("Connected to workspace.")
        except Exception as e:
            logger.error(f"Failed to connect to workspace: {str(e)}")
            raise
    
    # Get or create compute target for pipeline
    def get_or_create_compute_target(self):
       
        cluster_name = self.config['compute_cluster_name']
        try:
            self.compute_target = ComputeTarget(workspace=self.ws, name=cluster_name)
            logger.info("Found existing compute target.")
        except ComputeTargetException:
            logger.info("Creating a new compute target...")
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=self.config['vm_size'],
                max_nodes=self.config['max_nodes'],
                idle_seconds_before_scale_down=1800
            )
            self.compute_target = ComputeTarget.create(self.ws, cluster_name, compute_config)
            self.compute_target.wait_for_completion(show_output=True)
            logger.info("New compute target created.")
    
    
    #Create Azure ML environment for pipeline steps
    def create_environment(self):
        
        logger.info("Creating environment...")
        env = Environment('pipeline_environment')
        conda_dep = CondaDependencies.create(
            conda_packages=['pandas', 'scikit-learn', 'joblib', 'python=3.8'],
            pip_packages=['azureml-defaults', 'azureml-core', 'azureml-pipeline-core', 'azureml-pipeline-steps']
        )
        env.python.conda_dependencies = conda_dep
        return env

    # Create data processing pipeline step
    def create_data_processing_step(self, environment, datastore):
        
        logger.info("Creating data processing step...")
        processed_data = PipelineData("processed_data", datastore=datastore)
        data_processing_step = PythonScriptStep(
            name="Data Processing",
            script_name=__file__,  # Use this script as the source
            arguments=['--output-dir', processed_data],
            source_directory=".",
            compute_target=self.compute_target,
            runconfig=environment.get_run_config(os.path.basename(__file__)),
            outputs=[processed_data],
            allow_reuse=False
        )
        data_processing_step._script_name = "data_processing.py"  # Override to correctly log the script name
        return data_processing_step, processed_data

    #Create customer clustering pipeline step
    def create_clustering_step(self, environment, input_data, datastore):
        
        logger.info("Creating clustering step...")
        clustered_data = PipelineData("clustered_data", datastore=datastore)
        models_dir = PipelineData("models_dir", datastore=datastore)
        clustering_step = PythonScriptStep(
            name="Customer Clustering",
            script_name=__file__,
            arguments=['--input-dir', input_data, '--output-dir', clustered_data, '--models-dir', models_dir],
            source_directory=".",
            compute_target=self.compute_target,
            runconfig=environment.get_run_config(os.path.basename(__file__)),
            inputs=[input_data],
            outputs=[clustered_data, models_dir],
            allow_reuse=False
        )
        clustering_step._script_name = "clustering.py"
        return clustering_step, clustered_data, models_dir
    
    #Create model training pipeline step
    def create_model_training_step(self, environment, input_data, models_dir):
        
        logger.info("Creating model training step...")
        training_step = PythonScriptStep(
            name="Model Training",
            script_name=__file__,
            arguments=['--input-dir', input_data, '--models-dir', models_dir],
            source_directory=".",
            compute_target=self.compute_target,
            runconfig=environment.get_run_config(os.path.basename(__file__)),
            inputs=[input_data],
            outputs=[models_dir],
            allow_reuse=False
        )
        training_step._script_name = "model_training.py"
        return training_step
    
    #Create model evaluation pipeline step
    def create_evaluation_step(self, environment, models_dir, datastore):
       
        logger.info("Creating evaluation step...")
        evaluation_data = PipelineData("evaluation_data", datastore=datastore)
        evaluation_step = PythonScriptStep(
            name="Model Evaluation",
            script_name=__file__,
            arguments=['--models-dir', models_dir, '--output-dir', evaluation_data],
            source_directory=".",
            compute_target=self.compute_target,
            runconfig=environment.get_run_config(os.path.basename(__file__)),
            inputs=[models_dir],
            outputs=[evaluation_data],
            allow_reuse=False
        )
        evaluation_step._script_name = "evaluation.py"
        return evaluation_step, evaluation_data

    #Create model registration pipeline step
    def create_register_model_step(self, environment, models_dir, evaluation_data):
        
        logger.info("Creating model registration step...")
        register_step = PythonScriptStep(
            name="Register Model",
            script_name=__file__,
            arguments=['--models-dir', models_dir, '--evaluation-data', evaluation_data],
            source_directory=".",
            compute_target=self.compute_target,
            runconfig=environment.get_run_config(os.path.basename(__file__)),
            inputs=[models_dir, evaluation_data],
            allow_reuse=False
        )
        register_step._script_name = "register_model.py"
        return register_step

    #Build the complete pipeline
    def build_pipeline(self):
       
        logger.info("Building the complete pipeline...")
        datastore = self.ws.get_default_datastore()
        environment = self.create_environment()

        data_processing_step, processed_data = self.create_data_processing_step(environment, datastore)
        clustering_step, clustered_data, models_dir = self.create_clustering_step(environment, processed_data, datastore)
        training_step = self.create_model_training_step(environment, clustered_data, models_dir)
        evaluation_step, evaluation_data = self.create_evaluation_step(environment, models_dir, datastore)
        register_step = self.create_register_model_step(environment, models_dir, evaluation_data)

        # Connect the steps
        training_step.run_after(clustering_step)
        evaluation_step.run_after(training_step)
        register_step.run_after(evaluation_step)

        self.pipeline = Pipeline(workspace=self.ws, steps=[data_processing_step, clustering_step, training_step, evaluation_step, register_step])
        self.pipeline.validate()
        logger.info("Pipeline built and validated successfully.")

    #Run the pipeline
    def run_pipeline(self):
        
        if not self.pipeline:
            logger.error("Pipeline not built. Call build_pipeline() first.")
            return

        logger.info("Submitting pipeline to Azure ML...")
        self.experiment = Experiment(self.ws, "customer-churn-pipeline")
        pipeline_run = self.experiment.submit(self.pipeline)
        
        logger.info(f"Pipeline run submitted: {pipeline_run.get_portal_url()}")
        pipeline_run.wait_for_completion(show_output=True)
        return pipeline_run.get_published_pipeline()

    # Display metrics from pipeline run
    def display_pipeline_metrics(self, pipeline_run):
        
        
        try:
            evaluation_report_run = pipeline_run.find_step_run('Model Evaluation')
            evaluation_report = evaluation_report_run.get_outputs()[0].download_file()
            
            with open(evaluation_report, 'r') as f:
                metrics = json.load(f)
            
            logger.info("Pipeline run metrics:")
            for metric, value in metrics.items():
                logger.info(f"- {metric}: {value}")
        except Exception as e:
            logger.error(f"Could not retrieve metrics: {str(e)}")
    
    #Schedule pipeline to run periodically
    def schedule_pipeline(self, schedule_name: str, schedule_expression: str = "0 0 * * 0"):
        try:
            # Get published pipeline
            pipelines = self.ws.pipelines
            published_pipeline = list(pipelines.values())[0]  # Get first published pipeline
            
            # Create schedule
            from azureml.pipeline.core import ScheduleRecurrence, Schedule
            
            recurrence = ScheduleRecurrence(
                frequency="Week",
                interval=1,
                week_days=["Sunday"],
                time_of_day="00:00"
            )
            
            schedule = Schedule.create(
                self.ws,
                name=schedule_name,
                description="Weekly customer segmentation pipeline run",
                pipeline_id=published_pipeline.id,
                experiment_name="customer-segmentation-pipeline",
                recurrence=recurrence,
                wait_for_provisioning=True
            )
            
            logger.info(f"Pipeline scheduled: {schedule.name}")
            logger.info(f"Next run: {schedule.next_run_time}")
            
            return schedule
            
        except Exception as e:
            logger.error(f"Error scheduling pipeline: {str(e)}")
            raise

     #Clean up resources
    def cleanup(self):
        try:
            # Stop compute target to save costs
            if self.compute_target:
                self.compute_target.stop()
                logger.info(f"Compute target {self.compute_target.name} stopped")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="Run Customer Segmentation Pipeline")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file")
    
    args = parser.parse_args()
    
    # A simplified main function to demonstrate building and running
    # a self-contained pipeline.
    pipeline = None
    try:
        pipeline = CustomerSegmentationPipeline(config_path=args.config)
        pipeline.connect_to_workspace()
        pipeline.get_or_create_compute_target()
        pipeline.build_pipeline()
        pipeline_run = pipeline.run_pipeline()
        
        if pipeline_run:
            pipeline.display_pipeline_metrics(pipeline_run)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        
if __name__ == "__main__":
   
    main()
