
import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={"usage": 5, "tenure": 8, "tickets": 3}
    )

    assert response.status_code == 200
    assert 0 <= response.json()["churn_probability"] <= 1


@pytest.mark.parametrize("customer", [
    {"usage": -5, "tenure": 8, "tickets": 3},
    {"usage": 5, "tickets": 3},
    {"usage": "unknown", "tenure": 8, "tickets": 3},
    {"usage": 5, "tenure": 8, "tickets": -1}
])
def test_invalid_requests(customer):
    response = client.post("/predict", json=customer)
    assert response.status_code == 422
