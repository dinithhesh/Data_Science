"""
Utility Functions for Customer Segmentation Pipeline
Common helper functions and utilities for data processing, visualization, and file operations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import json
import joblib
import yaml
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
import warnings
from pathlib import Path
import pickle
import gzip
import tempfile
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class PipelineUtils:
    """Utility class for common pipeline operations"""
    
    @staticmethod
    def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> None:
        """
        Set up logging configuration
        
        Args:
            log_level: Logging level
            log_file: Path to log file (optional)
        """
        handlers = [logging.StreamHandler()]
        
        if log_file:
            handlers.append(logging.FileHandler(log_file))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        logger.info("Logging setup completed")

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON or YAML file
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                logger.warning(f"Config file {config_path} not found")
                return {}
            
            if config_path.suffix.lower() in ['.json']:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary
            config_path: Path to config file
        """
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_path.suffix.lower() in ['.json']:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            elif config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")

    @staticmethod
    def create_directory(directory_path: str) -> None:
        """
        Create directory if it doesn't exist
        
        Args:
            directory_path: Path to directory
        """
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {directory_path}")
        except Exception as e:
            logger.error(f"Error creating directory: {str(e)}")
            raise

    @staticmethod
    def save_data(df: pd.DataFrame, file_path: str, **kwargs) -> None:
        """
        Save DataFrame to various formats
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            **kwargs: Additional arguments for pandas save methods
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.suffix.lower() == '.csv':
                df.to_csv(file_path, index=kwargs.get('index', False))
            elif file_path.suffix.lower() == '.parquet':
                df.to_parquet(file_path, index=kwargs.get('index', False))
            elif file_path.suffix.lower() == '.pkl':
                df.to_pickle(file_path)
            elif file_path.suffix.lower() == '.json':
                df.to_json(file_path, orient=kwargs.get('orient', 'records'), 
                          indent=kwargs.get('indent', 2))
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Data saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    @staticmethod
    def load_data(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load DataFrame from various formats
        
        Args:
            file_path: Input file path
            **kwargs: Additional arguments for pandas load methods
            
        Returns:
            Loaded DataFrame
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                return pd.read_parquet(file_path, **kwargs)
            elif file_path.suffix.lower() == '.pkl':
                return pd.read_pickle(file_path)
            elif file_path.suffix.lower() == '.json':
                return pd.read_json(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    @staticmethod
    def save_model(model: Any, file_path: str, compress: bool = False) -> None:
        """
        Save trained model to file
        
        Args:
            model: Trained model object
            file_path: Output file path
            compress: Whether to compress the model file
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if compress:
                with gzip.open(f"{file_path}.gz", 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model saved (compressed) to: {file_path}.gz")
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model saved to: {file_path}")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @staticmethod
    def load_model(file_path: str, compressed: bool = False) -> Any:
        """
        Load trained model from file
        
        Args:
            file_path: Input file path
            compressed: Whether the model file is compressed
            
        Returns:
            Loaded model object
        """
        try:
            file_path = Path(file_path)
            
            if compressed:
                file_path = Path(f"{file_path}.gz")
            
            if not file_path.exists():
                raise FileNotFoundError(f"Model file not found: {file_path}")
            
            if compressed:
                with gzip.open(file_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
            
            logger.info(f"Model loaded from: {file_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def calculate_statistics(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for DataFrame
        
        Args:
            df: Input DataFrame
            columns: Specific columns to analyze (None for all numeric columns)
            
        Returns:
            Statistics dictionary
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {}
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'q1': float(df[col].quantile(0.25)),
                    'q3': float(df[col].quantile(0.75)),
                    'skewness': float(df[col].skew()),
                    'kurtosis': float(df[col].kurtosis()),
                    'missing_values': int(df[col].isnull().sum()),
                    'missing_percentage': float((df[col].isnull().sum() / len(df)) * 100),
                    'zeros': int((df[col] == 0).sum()),
                    'outliers': int(((df[col] - df[col].mean()).abs() > 3 * df[col].std()).sum())
                }
            else:
                stats[col] = {
                    'unique_values': int(df[col].nunique()),
                    'top_value': df[col].mode()[0] if not df[col].mode().empty else None,
                    'top_frequency': int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0,
                    'missing_values': int(df[col].isnull().sum()),
                    'missing_percentage': float((df[col].isnull().sum() / len(df)) * 100)
                }
        
        return stats

    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers in a column
        
        Args:
            df: Input DataFrame
            column: Column name to analyze
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            return z_scores > threshold
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")

    @staticmethod
    def plot_distribution(df: pd.DataFrame, column: str, figsize: tuple = (10, 6), 
                         output_path: Optional[str] = None) -> None:
        """
        Plot distribution of a column
        
        Args:
            df: Input DataFrame
            column: Column name to plot
            figsize: Figure size
            output_path: Path to save the plot
        """
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        if pd.api.types.is_numeric_dtype(df[column]):
            ax1.hist(df[column].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax1.set_title(f'Distribution of {column}')
            ax1.set_xlabel(column)
            ax1.set_ylabel('Frequency')
            
            # Boxplot
            ax2.boxplot(df[column].dropna())
            ax2.set_title(f'Boxplot of {column}')
            ax2.set_ylabel(column)
            
        else:
            # For categorical data
            value_counts = df[column].value_counts().head(10)
            ax1.bar(value_counts.index.astype(str), value_counts.values, alpha=0.7, edgecolor='black')
            ax1.set_title(f'Top 10 values in {column}')
            ax1.set_xlabel(column)
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            # Pie chart
            ax2.pie(value_counts.values, labels=value_counts.index.astype(str), autopct='%1.1f%%')
            ax2.set_title(f'Distribution of {column}')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Plot saved to: {output_path}")
        
        plt.show()

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                               figsize: tuple = (12, 10), output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Plot correlation matrix for numeric columns
        
        Args:
            df: Input DataFrame
            columns: Specific columns to include
            figsize: Figure size
            output_path: Path to save the plot
            
        Returns:
            Correlation matrix DataFrame
        """
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
        else:
            numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_columns) < 2:
            logger.warning("Not enough numeric columns for correlation matrix")
            return pd.DataFrame()
        
        corr_matrix = df[numeric_columns].corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, mask=mask)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Correlation matrix saved to: {output_path}")
        
        plt.show()
        
        return corr_matrix

    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get memory usage information for DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Memory usage information
        """
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        return {
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / 1024 ** 2,
            'per_column': memory_usage.to_dict(),
            'dtypes': df.dtypes.to_dict()
        }

    @staticmethod
    def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        df_optimized = df.copy()
        
        # Downcast numeric columns
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Convert object columns to category if beneficial
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized[col]) < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"Memory reduced by {reduction:.2f}% "
                   f"({original_memory / 1024**2:.2f} MB -> {optimized_memory / 1024**2:.2f} MB)")
        
        return df_optimized

    @staticmethod
    def generate_timestamp() -> str:
        """
        Generate current timestamp string
        
        Returns:
            Timestamp string in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def time_execution(func):
        """
        Decorator to measure function execution time
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            logger.info(f"Function {func.__name__} executed in {execution_time}")
            return result
        
        return wrapper

# Example usage and test functions
def test_utils():
    """Test utility functions"""
    utils = PipelineUtils()
    
    # Create test data
    test_data = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randint(0, 10, 100),
        'C': ['cat', 'dog'] * 50
    })
    
    # Test directory creation
    utils.create_directory('test_output')
    
    # Test data saving/loading
    utils.save_data(test_data, 'test_output/test_data.csv')
    loaded_data = utils.load_data('test_output/test_data.csv')
    
    # Test statistics calculation
    stats = utils.calculate_statistics(test_data)
    print("Statistics:", stats)
    
    # Test memory optimization
    optimized_data = utils.optimize_memory_usage(test_data)
    
    # Test plotting
    utils.plot_distribution(test_data, 'A', output_path='test_output/distribution.png')
    
    # Clean up
    import shutil
    shutil.rmtree('test_output', ignore_errors=True)
    
    print("All utility tests passed!")

if __name__ == "__main__":
    test_utils()