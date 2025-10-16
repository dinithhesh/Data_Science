
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os, sys
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import load_model, predict_flood_event

app = FastAPI(
    title="Flood Event Prediction API",
    description="Predict flood events based on environmental features",
    version="1.0"
)

# ---------- Input Schema ----------
class FloodInput(BaseModel):
    precip_1h: float
    slope: float
    elevation: float
    timestamp: float
    precip_24h: float
    impervious_frac: float


# ---------- Load Model ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = load_model(MODEL_PATH)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
def read_root():
    # Serve the HTML page
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))



# ---------- API Routes ----------
@app.get("/")
def read_root():
    # Serve the HTML page
     return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.post("/predict")
def predict(input_data: FloodInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        prediction = predict_flood_event(model, df)
        result = {
            "predicted_flood_event": int(prediction["predicted_flood_event"].iloc[0]),
            "probability": float(prediction.get("probability", [None])[0])
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
