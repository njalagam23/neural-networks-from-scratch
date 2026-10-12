
import json
from pathlib import Path

import joblib
import torch

from app.model import ChurnModel
from app.preprocessing import preprocess_customer


# Find project folder automatically
PROJECT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


class ChurnPredictor:

    def __init__(self):

        # Load model configuration
        with open(ARTIFACTS_DIR / "model_config.json") as f:
            self.config = json.load(f)

        # Create neural network architecture
        self.model = ChurnModel(
            hidden_size=self.config["hidden_size"]
        )

        # Load trained weights
        state_dict = torch.load(
            ARTIFACTS_DIR / "churn_model.pth",
            map_location="cpu",
            weights_only=True
        )

        self.model.load_state_dict(state_dict)

        # Disable dropout/training behavior
        self.model.eval()

        # Load training scaler
        self.scaler = joblib.load(
            ARTIFACTS_DIR / "churn_scaler.joblib"
        )

        # Load classification threshold
        self.threshold = self.config[
            "classification_threshold"
        ]

    def predict(self, usage, tenure, tickets):

        # Step 1: Preprocess customer information
        customer_tensor = preprocess_customer(
            usage=usage,
            tenure=tenure,
            tickets=tickets,
            scaler=self.scaler
        )

        # Step 2: Predict without training
        with torch.no_grad():
            logit = self.model(customer_tensor)
            probability = torch.sigmoid(logit).item()

        # Step 3: Apply classification threshold
        risk_level = (
            "High Churn Risk"
            if probability >= self.threshold
            else "Low Churn Risk"
        )

        return {
            "churn_probability": round(probability, 4),
            "risk_level": risk_level
        }
