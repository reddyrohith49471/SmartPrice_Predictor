from fastapi import FastAPI
from pydantic import BaseModel
import time
from src.model_services.load_model import ModelLoader
from src.model_services.predictor import Predictor

app = FastAPI()
loader = ModelLoader()
model = loader.get_model()

predictor = Predictor(model)


class PredictionInput(BaseModel):
    combined_text: str
    Value: float

@app.get("/")
def home():
    return {"message": "Price Prediction API is running"}

@app.post("/predict")
def predict(data: PredictionInput):
    start = time.time()
    price = predictor.predict(
        text=data.combined_text,
        value=data.Value
    )
    latency = (time.time()-start)*1000
    return {"predicted_price": price,
            "Latency_ms":f"{round(latency,2)} ms"}
