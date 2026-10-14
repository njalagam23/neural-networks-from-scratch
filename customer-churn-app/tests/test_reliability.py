
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app, raise_server_exceptions=False)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_invalid_input():
    response = client.post(
        "/predict",
        json={
            "usage": -5,
            "tenure": 8,
            "tickets": 3
        }
    )

    assert response.status_code == 422


def test_unexpected_error(monkeypatch):
    from app.api import predictor

    def broken_predict(**kwargs):
        raise RuntimeError("Internal test failure")

    monkeypatch.setattr(predictor, "predict", broken_predict)

    response = client.post(
        "/predict",
        json={
            "usage": 5,
            "tenure": 8,
            "tickets": 3
        }
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error"
    }
    assert "Internal test failure" not in response.text
