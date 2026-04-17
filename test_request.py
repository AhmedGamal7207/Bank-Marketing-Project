import requests

API_URL = "https://bank-marketing-project.onrender.com/predict"


def predict(
    age: int = 35,
    job: str = "management",
    marital: str = "married",
    education: str = "tertiary",
    balance: int = 1500,
    housing: str = "yes",
    loan: str = "no",
    contact: str = "cellular",
    month: str = "may",
    campaign: int = 2,
    previous: int = 0,
    poutcome: str = "unknown",
    pdays: int = -1,
) -> dict:
    """
    Send a prediction request to the deployed FastAPI model.

    Returns:
        dict with prediction (0/1), probability, and label
    """
    payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "campaign": campaign,
        "previous": previous,
        "poutcome": poutcome,
        "pdays": pdays,
    }

    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    # Example: a client likely to subscribe
    result = predict(
        age=28, job="student", marital="single", education="tertiary",
        balance=3000, housing="no", loan="no", contact="cellular",
        month="mar", campaign=1, previous=1, poutcome="success", pdays=100
    )
    print("Likely subscriber:", result)

    # Example: a client unlikely to subscribe
    result = predict(
        age=42, job="blue-collar", marital="married", education="primary",
        balance=200, housing="yes", loan="yes", contact="unknown",
        month="may", campaign=5, previous=0, poutcome="unknown", pdays=-1
    )
    print("Unlikely subscriber:", result)
