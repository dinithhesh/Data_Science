"""
Data Processing Module for Customer Segmentation and Churn Prediction
Handles data loading, cleaning, feature engineering, and RFM analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from io import StringIO
import logging
import os
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class to handle all data processing operations for customer data
    """
    
    def __init__(self, connection_string=None, container_name="customerdata", blob_name="marketing_campaign.csv"):
        """
        Initialize the data processor
        
        Args:
            connection_string (str): Azure Blob Storage connection string
            container_name (str): Azure container name
            blob_name (str): Blob file name
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_name = blob_name
        self.df = None
        self.rfm_df = None
        
    def load_data_from_azure_blob(self):
        """
        Load data from Azure Blob Storage
        
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            logger.info("Loading data from Azure Blob Storage...")
            
            # Create BlobServiceClient and container client
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            container_client = blob_service_client.get_container_client(self.container_name)
            
            # Download the blob file
            download_stream = container_client.download_blob(self.blob_name)
            data = download_stream.readall().decode('utf-8')
            
            # Load the CSV data into pandas DataFrame
            self.df = pd.read_csv(StringIO(data), sep='\t')
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data from Azure Blob: {str(e)}")
            raise
    
    def load_data_from_local(self, file_path="data/marketing_campaign.csv"):
        """
        Load data from local file for testing
        
        Args:
            file_path (str): Local file path
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            logger.info(f"Loading data from local file: {file_path}")
            self.df = pd.read_csv(file_path, sep='\t')
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading local file: {str(e)}")
            raise
    
    def clean_data(self):
        """
        Clean and preprocess the data
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data method first.")
        
        try:
            logger.info("Starting data cleaning...")
            
            # Original data info
            logger.info(f"Original data shape: {self.df.shape}")
            logger.info(f"Original missing values:\n{self.df.isnull().sum()}")
            
            # Drop rows with missing ID
            self.df = self.df.dropna(subset=['ID'])
            logger.info(f"After dropping missing IDs: {self.df.shape}")
            
            # Fill missing income with median
            if 'Income' in self.df.columns:
                income_median = self.df['Income'].median()
                self.df['Income'] = self.df['Income'].fillna(income_median)
                logger.info(f"Filled missing Income with median: {income_median}")
            
            # Convert date column
            if 'Dt_Customer' in self.df.columns:
                self.df['Dt_Customer'] = pd.to_datetime(self.df['Dt_Customer'], dayfirst=True, errors='coerce')
                # Drop rows with invalid dates
                self.df = self.df.dropna(subset=['Dt_Customer'])
                logger.info("Converted Dt_Customer to datetime")
            
            # Handle other potential missing values
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # For categorical columns, fill with mode
                    mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                    self.df[col] = self.df[col].fillna(mode_val)
                elif self.df[col].dtype in ['int64', 'float64']:
                    # For numerical columns, fill with median
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
            
            # Sort by ID for consistency
            self.df = self.df.sort_values(by="ID", ascending=True).reset_index(drop=True)
            
            logger.info(f"Final cleaned data shape: {self.df.shape}")
            logger.info(f"Final missing values:\n{self.df.isnull().sum()}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            raise
    
    def create_rfm_features(self):
        """
        Create RFM (Recency, Frequency, Monetary) features
        
        Returns:
            pd.DataFrame: RFM features DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data and clean_data methods first.")
        
        try:
            logger.info("Creating RFM features...")
            
            # Calculate snapshot date (max date + 1 day)
            snapshot_date = self.df['Dt_Customer'].max() + timedelta(days=1)
            logger.info(f"Snapshot date for RFM: {snapshot_date}")
            
            # Group by ID and calculate RFM metrics
            rfm = self.df.groupby('ID').agg({
                'Dt_Customer': lambda x: (snapshot_date - x.max()).days,  # Recency
                'ID': 'count',  # Frequency (assuming each row is a transaction)
                'Income': 'mean'  # Monetary value
            }).rename(columns={
                'Dt_Customer': 'Recency',
                'ID': 'Frequency',
                'Income': 'Monetary'
            })
            
            # Reset index to make ID a column
            rfm = rfm.reset_index()
            
            self.rfm_df = rfm
            logger.info(f"RFM features created. Shape: {self.rfm_df.shape}")
            logger.info(f"RFM stats:\n{self.rfm_df.describe()}")
            
            return self.rfm_df
            
        except Exception as e:
            logger.error(f"Error creating RFM features: {str(e)}")
            raise
    
    def create_churn_label(self, recency_threshold=90):
        """
        Create churn label based on recency threshold
        
        Args:
            recency_threshold (int): Days since last activity to consider as churn
            
        Returns:
            pd.DataFrame: RFM DataFrame with churn label
        """
        if self.rfm_df is None:
            raise ValueError("No RFM data. Call create_rfm_features first.")
        
        try:
            logger.info(f"Creating churn label with threshold: {recency_threshold} days")
            
            # Create churn label (1 = churn, 0 = active)
            self.rfm_df['churn'] = (self.rfm_df['Recency'] > recency_threshold).astype(int)
            
            # Log churn distribution
            churn_counts = self.rfm_df['churn'].value_counts()
            logger.info(f"Churn distribution:\n{churn_counts}")
            logger.info(f"Churn rate: {churn_counts[1] / len(self.rfm_df) * 100:.2f}%")
            
            return self.rfm_df
            
        except Exception as e:
            logger.error(f"Error creating churn label: {str(e)}")
            raise
    
    def prepare_model_data(self):
        """
        Prepare data for model training
        
        Returns:
            tuple: (X, y, feature_names) for model training
        """
        if self.rfm_df is None:
            raise ValueError("No RFM data. Call create_rfm_features first.")
        
        try:
            logger.info("Preparing data for model training...")
            
            # Select features and target
            X = self.rfm_df[['Recency', 'Frequency', 'Monetary']]
            y = self.rfm_df['churn'] if 'churn' in self.rfm_df.columns else None
            
            # Log feature statistics
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Features description:\n{X.describe()}")
            
            if y is not None:
                logger.info(f"Target shape: {y.shape}")
            
            return X, y, list(X.columns)
            
        except Exception as e:
            logger.error(f"Error preparing model data: {str(e)}")
            raise
    
    def scale_features(self, X):
        """
        Scale features using StandardScaler
        
        Args:
            X (pd.DataFrame): Features to scale
            
        Returns:
            tuple: (scaled_features, scaler_object)
        """
        try:
            logger.info("Scaling features...")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            logger.info("Features scaled successfully")
            return X_scaled, scaler
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise
    
    def save_processed_data(self, output_dir="processed_data"):
        """
        Save processed data to files
        
        Args:
            output_dir (str): Directory to save processed files
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save cleaned data
            if self.df is not None:
                cleaned_path = os.path.join(output_dir, "cleaned_marketing_campaign.csv")
                self.df.to_csv(cleaned_path, index=False)
                logger.info(f"Cleaned data saved to: {cleaned_path}")
            
            # Save RFM data
            if self.rfm_df is not None:
                rfm_path = os.path.join(output_dir, "rfm_analysis.csv")
                self.rfm_df.to_csv(rfm_path, index=False)
                logger.info(f"RFM data saved to: {rfm_path}")
            
            logger.info("All processed data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise
    
    def get_data_summary(self):
        """
        Get summary statistics of the processed data
        
        Returns:
            dict: Summary statistics
        """
        summary = {}
        
        if self.df is not None:
            summary['original_shape'] = self.df.shape
            summary['original_columns'] = list(self.df.columns)
            summary['original_missing_values'] = self.df.isnull().sum().to_dict()
        
        if self.rfm_df is not None:
            summary['rfm_shape'] = self.rfm_df.shape
            summary['rfm_stats'] = self.rfm_df[['Recency', 'Frequency', 'Monetary']].describe().to_dict()
            if 'churn' in self.rfm_df.columns:
                summary['churn_distribution'] = self.rfm_df['churn'].value_counts().to_dict()
        
        return summary

# Example usage and test function
def main():
    """Example usage of the DataProcessor class"""
    
    # Initialize processor
    processor = DataProcessor()
    
    try:
        # Load data (use local file for testing)
        df = processor.load_data_from_local("marketing_campaign.csv")
        
        # Clean data
        cleaned_df = processor.clean_data()
        
        # Create RFM features
        rfm_df = processor.create_rfm_features()
        
        # Create churn label
        rfm_with_churn = processor.create_churn_label(recency_threshold=90)
        
        # Prepare model data
        X, y, feature_names = processor.prepare_model_data()
        
        # Scale features
        X_scaled, scaler = processor.scale_features(X)
        
        # Save processed data
        processor.save_processed_data()
        
        # Get summary
        summary = processor.get_data_summary()
        print("Data processing completed successfully!")
        print(f"Summary: {summary}")
        
        return processor
        
    except Exception as e:
        print(f"Error in data processing: {e}")
        return None

if __name__ == "__main__":
    # Test the data processor
    processor = main()
    
    if processor is not None and processor.rfm_df is not None:
        print("\nFirst 5 rows of RFM data:")
        print(processor.rfm_df.head())
        
        print("\nChurn distribution:")
        print(processor.rfm_df['churn'].value_counts())