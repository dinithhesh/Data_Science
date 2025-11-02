from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import sys
from pathlib import Path

# Add src to Python path (so FastAPI can import predict.py)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict import load_model, predict_churn

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction Dashboard")

# Set up HTML template directory
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Load model and scaler once
MODEL_PATH = str(BASE_DIR / "models" / "random_forest_optimized.pkl")
SCALER_PATH = str(BASE_DIR / "models" / "scaler_20250921_021756.pkl")  # update if name differs

model, scaler = load_model(MODEL_PATH, SCALER_PATH)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render dashboard input form"""
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    recency: float = Form(...),
    frequency: float = Form(...),
    monetary: float = Form(...)
):
    """Predict churn probability"""
    customer_data = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }

    result_df = predict_churn(model, scaler, customer_data)
    churn_prob = float(result_df["churn_probability"].iloc[0])
    label = int(result_df["predicted_label"].iloc[0])

    label_text = "Likely to Churn 😞" if label == 1 else "Not Likely to Churn 😊"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": {
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "churn_prob": round(churn_prob, 4),
            "label": label_text
        }
    })
