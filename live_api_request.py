import os

import requests


API_URL = os.environ.get(
    "LIVE_API_URL",
    "https://nd0821-c3-starter-code-cfx8.onrender.com/predict",
)

PAYLOAD = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


response = requests.post(API_URL, json=PAYLOAD, timeout=30)

try:
    data = response.json()
    prediction = data.get("prediction")
except ValueError:
    prediction = response.text

print(f"HTTP status code: {response.status_code}")
print(f"Inference result: {prediction}")
