import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from starter.ml import model
from starter.ml.data import process_data


def make_dummy_data():
    df = pd.DataFrame(
        {
            "age": [25, 40, 50, 22],
            "workclass": [
                "Private",
                "Self-emp",
                "Private",
                "Private",
            ],
            "education": [
                "Bachelors",
                "HS-grad",
                "HS-grad",
                "Bachelors",
            ],
            "marital-status": [
                "Never-married",
                "Married",
                "Married",
                "Never-married",
            ],
            "occupation": [
                "Tech-support",
                "Exec-managerial",
                "Adm-clerical",
                "Sales",
            ],
            "relationship": [
                "Not-in-family",
                "Husband",
                "Husband",
                "Own-child",
            ],
            "race": [
                "White",
                "Black",
                "White",
                "White",
            ],
            "sex": [
                "Male",
                "Female",
                "Female",
                "Male",
            ],
            "capital-gain": [0, 1000, 0, 0],
            "capital-loss": [0, 0, 0, 0],
            "hours-per-week": [40, 50, 30, 20],
            "native-country": [
                "United-States",
                "United-States",
                "Canada",
                "United-States",
            ],
            "salary": [0, 1, 0, 0],
        }
    )
    return df


def test_train_and_inference_roundtrip(tmp_path):
    df = make_dummy_data()
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    clf = model.train_model(X, y)
    preds = model.inference(clf, X)
    assert preds.shape[0] == y.shape[0]
    # predictions should be 0/1
    assert set(np.unique(preds)).issubset({0, 1})


def test_train_model_and_inference_return_expected_types():
    df = make_dummy_data()
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    clf = model.train_model(X, y)
    preds = model.inference(clf, X[:3])

    assert isinstance(clf, RandomForestClassifier)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (3,)
    assert np.issubdtype(preds.dtype, np.integer)


def test_inference_predicts_zero_for_low_income_example():
    df = make_dummy_data()
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    clf = model.train_model(X, y)

    preds = model.inference(clf, X[:1])
    assert preds.tolist() == [0]


def test_inference_predicts_one_for_high_income_example():
    df = make_dummy_data()
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    clf = model.train_model(X, y)

    preds = model.inference(clf, X[1:2])
    assert preds.tolist() == [1]


def test_save_model(tmp_path):
    df = make_dummy_data()
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    clf = model.train_model(X, y)
    out = tmp_path / "out_model.joblib"
    encoder_out = tmp_path / "encoder.joblib"
    lb_out = tmp_path / "lb.joblib"
    path = model.save_model(
        clf,
        path=str(out),
        encoder=encoder,
        lb=lb,
        encoder_path=str(encoder_out),
        lb_path=str(lb_out),
    )
    assert path == str(out)
    assert out.exists()
    assert encoder_out.exists()
    assert lb_out.exists()

    loaded_model, loaded_encoder, loaded_lb = model.load_model_artifacts(
        model_path=str(out),
        encoder_path=str(encoder_out),
        lb_path=str(lb_out),
    )
    assert loaded_model is not None
    assert loaded_encoder is not None
    assert loaded_lb is not None


def test_evaluate_slices_returns_entries():
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = model.compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)

    slice_df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "label": [1, 1, 0, 0],
        }
    )
    X, y_slice, encoder, lb = process_data(
        slice_df,
        categorical_features=["group"],
        label="label",
        training=True,
    )
    clf = model.train_model(X, y_slice)
    results = model.evaluate_slices(
        clf,
        slice_df,
        categorical_features=["group"],
        label="label",
        encoder=encoder,
        lb=lb,
    )

    assert isinstance(results, dict)
    assert len(results) == 2
    assert results["group=A"] == pytest.approx((1.0, 1.0, 1.0))
    assert results["group=B"] == pytest.approx((1.0, 1.0, 1.0))


def test_write_slice_metrics_writes_output_file(tmp_path):
    df = make_dummy_data()
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    clf = model.train_model(X, y)

    output_path = tmp_path / "slice_output.txt"
    results = model.write_slice_metrics(
        clf,
        df,
        categorical_feature="education",
        label="salary",
        output_path=str(output_path),
        encoder=encoder,
        lb=lb,
    )

    assert isinstance(results, dict)
    assert output_path.exists()
    assert any(
        "education=" in line for line in output_path.read_text().splitlines()
    )


def test_training_script_accepts_data_path_and_saves_artifacts(tmp_path):
    from starter.train_model import main

    data_path = "data/census.csv"
    output_dir = tmp_path / "artifacts"

    result = main(
        data_path=data_path,
        output_dir=str(output_dir),
        test_size=0.2,
        random_state=42,
    )

    assert result["metrics"]["precision"] == pytest.approx(0.7418639053254438)
    assert result["metrics"]["recall"] == pytest.approx(0.6384468491406747)
    assert result["metrics"]["fbeta"] == pytest.approx(0.6862812179267875)
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "encoder.joblib").exists()
    assert (output_dir / "lb.joblib").exists()
