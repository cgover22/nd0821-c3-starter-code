from fastapi.testclient import TestClient
import os
from starter.api import app


client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Census prediction API"}


def make_payload(age):
    return {
        "age": age,
        "workclass": "Private",
        "fnlgt": 12345,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }


def test_post_predict_lower():
    # Use deterministic fallback rule during tests
    os.environ["DETERMINISTIC"] = "1"
    payload = make_payload(age=30)
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"


def test_post_predict_higher():
    os.environ["DETERMINISTIC"] = "1"
    payload = make_payload(age=70)
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] == ">50K"


def test_post_predict_real_model_path():
    os.environ.pop("DETERMINISTIC", None)
    payload = make_payload(age=39)
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] in {">50K", "<=50K"}


def test_openapi_includes_example_payload():
    schema = app.openapi()
    payload = schema["components"]["schemas"]["CensusIn"]["example"]

    assert payload["age"] == 39
    assert payload["workclass"] == "State-gov"
    assert payload["fnlgt"] == 77516
    assert payload["education"] == "Bachelors"
    assert payload["native-country"] == "United-States"
