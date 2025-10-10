import os
import pandas as pd

def load_data(filename: str = "colombo_flood_balanced_1000.csv") -> pd.DataFrame:
    """
    Load dataset from the data folder.

    Args:
        filename (str): Name of the CSV file inside the data folder.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
    data_path = os.path.join(base_dir, "data", filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    return pd.read_csv(data_path)
