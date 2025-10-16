# pipeline.py
# Add your data science pipeline steps here
import os
import sys


BASE_DIR = "/content/drive/MyDrive/flood-detect"
SRC_PATH = os.path.join(BASE_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


from data_loader import load_data
from eda import run_eda      # (You can define run_eda in src/eda.py)
from train import train_model  # (Define in src/train.py)
from predict import load_model, predict_flood_event

def run_pipeline():
    print("🚀 Starting pipeline...")

    # Step 1: Load dataset
    print("📂 Loading dataset...")
    df = load_data()
    print(f"✅ Data loaded successfully. Shape: {df.shape}")

    # Step 2: Run EDA
    print("📊 Running EDA...")
    results = run_eda(df, save_dir=os.path.join(BASE_DIR, "eda_outputs"))
    print("\n🎯 EDA Summary:")
    for key, value in results.items():
        print(f"{key}: {value}")

    print("\n✅ EDA pipeline completed successfully.")

   
     # Step 3: Train models
    model, results = train_model(df)

    print("\n🏁 Training completed. Best model and results summary:")
    print(results)

   # === Step 4: Load best model and run sample prediction ===
    print("🔍 Running test prediction...")
    model_path = os.path.join(BASE_DIR , "models", "model.pkl")
    model = load_model(model_path)

    sample = df.sample(5, random_state=42)
    predictions = predict_flood_event(model, sample)

    print("\n✅ Sample Predictions:")
    print(predictions[["precip_1h", "slope", "elevation", "predicted_flood_event"]])

    print("\n🏁 Pipeline Completed Successfully.")

if __name__ == "__main__":
    run_pipeline()
