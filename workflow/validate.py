import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from workflow.etl import load_data, split_data
from workflow.feature_selection import bin_age, calculate_family_size, calculate_fare_per_person, get_top_features


def prepare_test_data(model, X_test):
    # Transform X_test using the fitted pipeline
    X_test_processed = model.named_steps["preprocessor"].transform(X_test)
    # Convert transformed data to DataFrames for easier manipulation
    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=model.named_steps["preprocessor"].get_feature_names_out(),
    )
    return X_test_processed_df


def validate_model(X_test=None, y_test=None):
    # Load the test data if not provided
    if X_test is None or y_test is None:
        X, y = load_data()
        _, X_test, _, y_test = split_data(X, y)

    # Prepare the test data
    etl_model = joblib.load("models/etl-model.pkl")
    X_test = prepare_test_data(etl_model, X_test)
    X_test = calculate_family_size(X_test)
    X_test = calculate_fare_per_person(X_test)
    X_test = bin_age(X_test)
    top_features = get_top_features(etl_model)
    X_test = X_test[top_features]

    # Make predictions on the test set
    classifier = joblib.load("models/classifier.pkl")
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]  # For AUC score

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "auc": roc_auc,
    }
