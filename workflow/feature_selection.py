import joblib
import pandas as pd

from workflow.core import N_FEATURES


def calculate_family_size(data):
    # Family size as a new feature
    data["family_size"] = data["num__sibsp"] + data["num__parch"]
    return data


def calculate_fare_per_person(data):
    # Fare per person (fare divided by family size + 1 to avoid division by zero)
    data["fare_per_person"] = data["num__fare"] / (data["family_size"] + 1)
    return data


def bin_age(data):
    # Age group - Binning age into categories
    age_bins = [0, 12, 18, 30, 50, 80]
    age_labels = ["Child", "Teen", "Young Adult", "Adult", "Senior"]
    data["age_group"] = pd.cut(data["num__age"], bins=age_bins, labels=age_labels)
    # Apply one-hot encoding to age group
    data = pd.get_dummies(data, columns=["age_group"], drop_first=True)
    return data


def get_top_features(model):
    # Access feature importances from the RandomForestClassifier
    importances = model.named_steps["classifier"].feature_importances_

    # Ensure we get the correct feature names from the preprocessor
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # Check that the lengths match
    if len(importances) != len(feature_names):
        print(f"Length mismatch: {len(importances)} importances vs {len(feature_names)} features")

    # Create a DataFrame with feature names and their importances
    feature_importances = pd.DataFrame({"feature": feature_names, "importance": importances})

    # Sort features by importance
    feature_importances = feature_importances.sort_values(by="importance", ascending=False)
    return feature_importances.head(N_FEATURES)["feature"].values


def select_features(X_train):
    etl_model = joblib.load("models/etl-model.pkl")
    X_train = calculate_family_size(X_train)
    X_train = calculate_fare_per_person(X_train)
    X_train = bin_age(X_train)
    top_features = get_top_features(etl_model)
    return X_train[top_features]
