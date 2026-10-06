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


def test_post_predict_lower_with_real_model():
    os.environ.pop("DETERMINISTIC", None)
    app.state.model = None
    app.state.encoder = None
    app.state.lb = None

    payload = make_payload(age=20)
    payload["hours-per-week"] = 20
    payload["capital-gain"] = 0
    payload["capital-loss"] = 0
    payload["education"] = "Bachelors"
    payload["marital-status"] = "Never-married"

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"


def test_post_predict_higher_with_real_model():
    os.environ.pop("DETERMINISTIC", None)
    app.state.model = None
    app.state.encoder = None
    app.state.lb = None

    payload = make_payload(age=52)
    payload["education"] = "Masters"
    payload["education-num"] = 14
    payload["marital-status"] = "Married-civ-spouse"
    payload["occupation"] = "Exec-managerial"
    payload["relationship"] = "Husband"
    payload["capital-gain"] = 5000
    payload["capital-loss"] = 0
    payload["hours-per-week"] = 60

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] == ">50K"


def test_openapi_includes_example_payload():
    schema = app.openapi()
    payload = schema["components"]["schemas"]["CensusIn"]["example"]

    assert payload["age"] == 39
    assert payload["workclass"] == "State-gov"
    assert payload["fnlgt"] == 77516
    assert payload["education"] == "Bachelors"
    assert payload["native-country"] == "United-States"
