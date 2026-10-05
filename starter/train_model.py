"""Train and persist the Census income model using the project helpers."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.ml.data import process_data  # noqa: E402
from starter.ml.model import (  # noqa: E402
    compute_model_metrics,
    inference,
    save_model,
    train_model,
    write_slice_metrics,
)


DEFAULT_CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def train_and_evaluate_model(
    df,
    categorical_features=None,
    label="salary",
    test_size=0.2,
    random_state=42,
):
    """Split data, train the model, and return evaluation metrics."""
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=categorical_features,
        label=label,
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "encoder": encoder,
        "lb": lb,
        "model": model,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        },
    }


def main(
    data_path="data/census.csv",
    output_dir="model",
    categorical_features=None,
    label="salary",
    test_size=0.2,
    random_state=42,
):
    """Train the model from a CSV file and save the model plus
    preprocessor artifacts.
    """
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    df = pd.read_csv(data_path)
    result = train_and_evaluate_model(
        df,
        categorical_features=categorical_features,
        label=label,
        test_size=test_size,
        random_state=random_state,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = str(output_path / "model.joblib")
    encoder_path = str(output_path / "encoder.joblib")
    lb_path = str(output_path / "lb.joblib")

    save_model(
        result["model"],
        path=model_path,
        encoder=result["encoder"],
        lb=result["lb"],
        encoder_path=encoder_path,
        lb_path=lb_path,
    )

    return {
        "model_path": model_path,
        "encoder_path": encoder_path,
        "lb_path": lb_path,
        "metrics": result["metrics"],
        "train_size": len(result["train_df"]),
        "test_size": len(result["test_df"]),
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Census income model."
    )
    parser.add_argument(
        "--data-path",
        default="data/census.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="model",
        help="Directory for saved model artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the data to reserve for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for train/test split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = pd.read_csv(args.data_path)
    result = train_and_evaluate_model(
        df,
        categorical_features=DEFAULT_CATEGORICAL_FEATURES,
        label="salary",
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_model(
        result["model"],
        path=str(Path(args.output_dir) / "model.joblib"),
        encoder=result["encoder"],
        lb=result["lb"],
        encoder_path=str(Path(args.output_dir) / "encoder.joblib"),
        lb_path=str(Path(args.output_dir) / "lb.joblib"),
    )
    slice_output_path = Path("slice_output.txt")
    slice_results = write_slice_metrics(
        result["model"],
        df,
        categorical_feature="education",
        label="salary",
        output_path=slice_output_path,
        categorical_features=DEFAULT_CATEGORICAL_FEATURES,
        encoder=result["encoder"],
        lb=result["lb"],
    )
    print(f"Saved slice metrics to {slice_output_path}")
    for key, (precision, recall, fbeta) in slice_results.items():
        print(
            f"{key}: precision={precision:.4f}, "
            f"recall={recall:.4f}, fbeta={fbeta:.4f}"
        )
    print(result["metrics"])
