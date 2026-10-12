
from app.predict import ChurnPredictor


def main():
    predictor = ChurnPredictor()

    customer = {
        "usage": 5,
        "tenure": 8,
        "tickets": 3
    }

    result = predictor.predict(**customer)

    print("Customer:", customer)
    print("Prediction:", result)


if __name__ == "__main__":
    main()
