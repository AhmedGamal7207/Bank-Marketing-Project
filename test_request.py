import requests

API_URL = "https://bank-marketing-project.onrender.com/predict"


def predict(dummy_parm1: float, dummy_parm2: float, dummy_parm3: float) -> dict:
    """
    Send a prediction request to the deployed FastAPI model.

    Args:
        dummy_parm1: First input feature (float)
        dummy_parm2: Second input feature (float)
        dummy_parm3: Third input feature (float)

    Returns:
        dict with key "prediction" containing the model's output
    """
    payload = {
        "dummy_parm1": dummy_parm1,
        "dummy_parm2": dummy_parm2,
        "dummy_parm3": dummy_parm3,
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    result = predict(1.0, 2.0, 3.0)
    print("Prediction:", result)
