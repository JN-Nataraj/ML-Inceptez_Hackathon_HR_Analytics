from fastapi import FastAPI, HTTPException
import joblib as jbl
from pydantic import BaseModel
import pandas as pd
from typing import Optional
from src.featureengineering import FeatureEngineering


app = FastAPI()

model = None
threshold = 0.5

def Load_Model():
    global model
    global threshold
    package = jbl.load("Models/hr_analytics_thrsh77_model.pkl")
    model = package['model']
    threshold = package['threshold']

    return model

@app.on_event("startup")
def startup_event():
    Load_Model()
    print("Model Loaded Successfully")

class Predict_Request(BaseModel):
    employee_id: int
    department: str
    region: str
    education: str
    gender: str
    recruitment_channel: str
    no_of_trainings: int
    age: int
    previous_year_rating: Optional[float] = None
    length_of_service: int
    KPIs_met_80: int
    awards_won: int
    avg_training_score: int


class Predict_Response(BaseModel):
    employee_id: int
    promotion_prediction_class: str
    promotion_prediction: int
    probability: float

@app.post("/predict", response_model=Predict_Response)
def predict_promotion(request: Predict_Request) -> Predict_Response:
    try:
        print("Received prediction request:", request)
        if model is None:
            print("Model is not loaded successfully")
        else:
            print("Model is loaded, proceeding with prediction")
            input_data = request.dict()
            employee_id = input_data.pop('employee_id')
            data_df = pd.DataFrame([input_data])
            data_df = data_df.rename(columns={"KPIs_met_80": "KPIs_met >80%"})
            data_df = data_df.rename(columns={"awards_won": "awards_won?"})
            promotion_prediction = model.predict_proba(data_df)[0,1]
            print("Best Threshold:", threshold)
            promotion_prediction_class = "Promoted" if promotion_prediction >= threshold else "Not Promoted"

            return Predict_Response(
                        employee_id=employee_id,
                        promotion_prediction_class=promotion_prediction_class,
                        promotion_prediction=int(promotion_prediction >= threshold),
                        probability=promotion_prediction
            )
    except Exception as e:
        print("ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))






        



    