"""Script to train machine learning model.

This file is intentionally lightweight for the starter repo and is not
invoked by the test suite. It provides an example of how to train a model
when run manually.
"""

from sklearn.model_selection import train_test_split

try:
    # Local imports that may not be present in the test environment
    from starter.ml.data import process_data
    from starter.ml.model import train_model, save_model
    import pandas as pd
except Exception:
    # If dependencies aren't available (e.g., during lint-only CI), keep
    # this module import-safe and exit early.
    raise SystemExit("train_model requires the project dependencies and data to run")


# flake8: noqa

def main():
    df = pd.read_csv("data/census.csv")
    train, _ = train_test_split(df, test_size=0.20)

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

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    clf = train_model(X_train, y_train)
    save_model(
        clf,
        path="model/model.joblib",
        encoder=encoder,
        lb=lb,
    )


if __name__ == "__main__":
    main()
