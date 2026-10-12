
import pytest

from app.predict import ChurnPredictor


@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


def test_prediction_probability(predictor):
    result = predictor.predict(
        usage=5,
        tenure=8,
        tickets=3
    )

    assert 0 <= result["churn_probability"] <= 1


@pytest.mark.parametrize(
    "usage, tenure, tickets, expected_risk",
    [
        (5, 8, 3, "High Churn Risk"),
        (30, 48, 0, "Low Churn Risk"),
        (12, 15, 2, "High Churn Risk"),
    ]
)
def test_customer_risk(
    predictor, usage, tenure, tickets, expected_risk
):
    result = predictor.predict(usage, tenure, tickets)

    assert result["risk_level"] == expected_risk


@pytest.mark.parametrize(
    "usage, tenure, tickets",
    [
        (-5, 8, 3),
        ("unknown", 8, 3),
        (5, -2, 3),
        (5, 8, -1),
    ]
)
def test_invalid_customer_inputs(
    predictor, usage, tenure, tickets
):
    with pytest.raises(ValueError):
        predictor.predict(usage, tenure, tickets)


def test_model_loading(predictor):

    # The model should be in evaluation mode
    assert predictor.model.training is False

    # Verify model architecture
    assert predictor.model.hidden.in_features == 3
    assert predictor.model.output.out_features == 1


def test_prediction_consistency(predictor):

    # Predict twice for the same customer
    result1 = predictor.predict(5, 8, 3)
    result2 = predictor.predict(5, 8, 3)

    # Results should match
    assert result1 == result2
