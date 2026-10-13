
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.predict import ChurnPredictor

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

predictor = ChurnPredictor()


class CustomerInput(BaseModel):
    usage: float = Field(ge=0)
    tenure: int = Field(ge=0)
    tickets: int = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    return predictor.predict(
        usage=customer.usage,
        tenure=customer.tenure,
        tickets=customer.tickets
    )
