# Model Card

**Model:** Census income classifier (RandomForest)

**Version:** 0.1

**Model Details:**
- **Developed by:** Exercise starter code for the Udacity ML ops project
- **Model type:** `sklearn.ensemble.RandomForestClassifier` trained with 100 estimators and `random_state=42`.
- **Date trained:** 2026-09-27

**Intended Use:**
- This model predicts whether an individual's annual income exceeds $50K (binary classification) from U.S. Census-derived features. It is intended for educational and demonstration purposes only and not for production decision-making without additional validation.

**Training Data:**
- The model was trained on the cleaned census dataset provided in [data/census.csv](data/census.csv). The training set contained 26,048 examples when run with the project defaults.

**Evaluation Data:**
- Evaluation was performed on a holdout test split from the same CSV dataset (20% test split). The test set contained 6,513 examples for the run reported below.

**Metrics:**
- **Metrics used:** Precision, recall, and F1-score (implemented as `fbeta` with `beta=1`).
- **Reported performance:** On the holdout test split the model achieved precision = 0.7419, recall = 0.6384, and F1 = 0.6863.
- These metrics were computed using `sklearn.metrics.precision_score`, `recall_score`, and `fbeta_score` (with `zero_division=1`) on binarized predictions from the trained model.

**Ethical Considerations:**
- The training data reflects historical census patterns and may contain biases correlated with protected attributes such as `race`, `sex`, and `native-country`. These biases can be learned by the model and lead to disparate performance across groups.
- Do not use this model for high-stakes or legally regulated decisions without a thorough fairness assessment, stakeholder review, and appropriate mitigation strategies.

**Caveats and Recommendations:**
- This model is provided as educational starter code and should not be deployed without additional validation, calibration, and monitoring.
- Before any deployment: perform fairness audits (slice-based metrics are supported by the repository), assess calibration, and evaluate robustness to distribution shift.
- Retrain frequently with updated, representative data and consider additional features or model architectures if higher recall or precision is required for the use case.

**Reproducibility notes:**
- The repository includes `starter/train_model.py` which trains the model and saves artifacts to the `model/` directory. Running the training script with the default parameters produced the metrics reported above and saved `model.joblib`, `encoder.joblib`, and `lb.joblib`.

