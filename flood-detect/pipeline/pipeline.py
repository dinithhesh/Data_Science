# pipeline.py
# Add your data science pipeline steps here
import os
import sys

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_data
from eda import run_eda      # (You can define run_eda in src/eda.py)
from train import train_model  # (Define in src/train.py)


def run_pipeline():
    print("🚀 Starting pipeline...")

    # Step 1: Load dataset
    print("📂 Loading dataset...")
    df = load_data()
    print(f"✅ Data loaded successfully. Shape: {df.shape}")

    # Step 2: Run EDA
    print("📊 Running EDA...")
    results = run_eda(df, save_dir=os.path.join(os.path.dirname(__file__), "eda_outputs"))
    print("\n🎯 EDA Summary:")
    for key, value in results.items():
        print(f"{key}: {value}")

    print("\n✅ EDA pipeline completed successfully.")

    # Step 3: Train model
    print("🤖 Training model...")
    model = train_model(df)

    print("🎉 Pipeline completed successfully.")
    return model


if __name__ == "__main__":
    run_pipeline()
