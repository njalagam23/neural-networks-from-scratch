
import numpy as np
import torch


def preprocess_customer(usage, tenure, tickets, scaler):

    # Validate customer inputs
    values = [usage, tenure, tickets]

    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Inputs must be numeric")

        if not isinstance(value, (int, float, np.number)):
            raise ValueError("Inputs must be numeric")

        if not np.isfinite(value):
            raise ValueError("Inputs must be finite")

        if value < 0:
            raise ValueError("Inputs cannot be negative")

    # Create customer feature array
    customer = np.array(
        [[usage, tenure, tickets]],
        dtype=np.float64
    )

    # Apply the scaler fitted during training
    customer_scaled = scaler.transform(customer)

    # Convert to PyTorch tensor
    customer_tensor = torch.tensor(
        customer_scaled,
        dtype=torch.float32
    )

    return customer_tensor
