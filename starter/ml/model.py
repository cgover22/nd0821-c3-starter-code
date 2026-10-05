from sklearn.metrics import fbeta_score, precision_score, recall_score
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# flake8: noqa

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    # Simple RandomForest classifier with sensible defaults
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(
    model,
    path="model/model.joblib",
    encoder=None,
    lb=None,
    encoder_path=None,
    lb_path=None,
):
    """Save trained model and optional preprocessing artifacts to disk.

    Creates parent directories if needed and saves the fitted encoder and label
    binarizer alongside the model when provided. Returns the model path.
    """
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    joblib.dump(model, path)

    if encoder is not None:
        enc_path = encoder_path or os.path.join(dirpath, "encoder.joblib")
        joblib.dump(encoder, enc_path)
    if lb is not None:
        lb_file_path = lb_path or os.path.join(dirpath, "lb.joblib")
        joblib.dump(lb, lb_file_path)

    return path


def load_model_artifacts(
    model_path="model/model.joblib",
    encoder_path="model/encoder.joblib",
    lb_path="model/lb.joblib",
):
    """Load the saved model and its preprocessing artifacts from disk."""
    model_obj = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
    return model_obj, encoder, lb


def evaluate_slices(
    model,
    data: pd.DataFrame,
    categorical_features,
    label,
    encoder=None,
    lb=None,
):
    """Evaluate model performance on slices of the data for each categorical feature.

    Returns a dict mapping "feature=value" -> (precision, recall, fbeta)
    """
    results = {}
    # If encoder/lb provided, use process_data from sibling module to transform
    from starter.ml.data import process_data

    # If encoder/lb not provided we need to fit them on the whole data
    if encoder is None or lb is None:
        _, _, encoder, lb = process_data(
            data,
            categorical_features=categorical_features,
            label=label,
            training=True,
        )

    for cat in categorical_features:
        values = data[cat].dropna().unique()
        for val in values:
            slice_df = data[data[cat] == val]
            if slice_df.shape[0] == 0:
                continue
            X_slice, y_slice, _, _ = process_data(
                slice_df,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            if X_slice.shape[0] == 0:
                continue
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            results[f"{cat}={val}"] = (precision, recall, fbeta)
    return results


def write_slice_metrics(
    model,
    data: pd.DataFrame,
    categorical_feature,
    label,
    output_path="slice_output.txt",
    categorical_features=None,
    encoder=None,
    lb=None,
):
    """Compute and write model metrics for each slice of one categorical feature.

    The output file contains one line per category value in the form
    "feature=value: precision=..., recall=..., fbeta=...".
    """
    if categorical_features is None:
        categorical_features = [
            col
            for col in data.select_dtypes(include=["object", "category", "string"]).columns
            if col != label
        ]
    elif isinstance(categorical_features, str):
        categorical_features = [categorical_features]

    all_results = evaluate_slices(
        model,
        data,
        categorical_features=categorical_features,
        label=label,
        encoder=encoder,
        lb=lb,
    )
    results = {key: value for key, value in all_results.items() if key.startswith(f"{categorical_feature}=")}

    output_path = str(output_path)
    dirpath = os.path.dirname(output_path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    lines = []
    for key, (precision, recall, fbeta) in results.items():
        lines.append(
            f"{key}: precision={precision:.4f}, recall={recall:.4f}, fbeta={fbeta:.4f}"
        )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
        if lines:
            file.write("\n")

    return results
