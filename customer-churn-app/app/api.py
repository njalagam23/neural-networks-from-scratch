from fastapi.responses import JSONResponse
import logging
import time
from fastapi import Request

from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.predict import ChurnPredictor

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "method=%s path=%s status=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            status_code,
            duration_ms
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


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "customer-churn-api"
    }


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception at %s",
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__)
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
