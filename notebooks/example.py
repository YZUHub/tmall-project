import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the Titanic dataset from Seaborn
data = sns.load_dataset("titanic")

# Drop rows with missing values in the target column ('survived') for simplicity
data = data.dropna(subset=["survived"])

# Define target and features
X = data.drop("survived", axis=1)
y = data["survived"]

# Split into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and categorical feature groups
numerical_features = ["age", "sibsp", "parch", "fare"]
categorical_features = ["pclass", "sex", "embarked", "who", "adult_male", "alone"]
drop_features = ["deck", "embark_town", "class", "alive"]

X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Numerical pipeline: impute missing values with median
numerical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Categorical pipeline: impute with mode and one-hot encode
categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent", fill_value="missing")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]
)

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    [
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
        ("drop_cols", "drop", drop_features),  # Drop unwanted columns
    ]
)


def train_grids(algorithm, model_params, grid_params, X_train, X_test, y_train, y_test):
    # Full pipeline with preprocessor and model
    model_pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", algorithm(**model_params))]  # Substitute with any other classifier as needed
    )

    # Define the parameter grid for grid search
    preprocessor_params = {
        "preprocessor__num__imputer__strategy": ["mean", "median"],  # Options for numerical imputation
        "preprocessor__cat__imputer__strategy": ["most_frequent", "constant"],  # Options for categorical imputation
    }
    algorithm_params = {f"classifier__{key}": value for key, value in grid_params.items()}
    search_grid_params = {**preprocessor_params, **algorithm_params}

    # Setup GridSearchCV with the pipeline and parameter grid
    grid_search = GridSearchCV(model_pipeline, search_grid_params, cv=5, scoring="accuracy", refit=True, n_jobs=-1)

    # Fit the model on the training data
    grid_search.fit(X_train, y_train)

    # Output the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Access the best pipeline with optimized parameters
    best_pipeline = grid_search.best_estimator_

    # Evaluate the optimized pipeline on the test data
    test_score = best_pipeline.score(X_test, y_test)
    print("Test Score with Best Pipeline:", test_score)

    return best_pipeline


# RandomForestClassifier run
grid_params = {
    "n_estimators": [20, 30, 40, 50, 60],  # Number of trees in the forest
    "max_depth": [3, 4, 5, 6],  # Maximum depth of the tree
    "max_features": [3, 5, 10],
}
model_params = {"random_state": 42}

random_forest_clf = train_grids(
    RandomForestClassifier,
    model_params,
    grid_params,
    X_train_imp,
    X_test_imp,
    y_train_imp,
    y_test_imp,
)

# Transform X_train and X_test using the fitted pipeline
X_train_processed = random_forest_clf.named_steps["preprocessor"].transform(X_train)
X_test_processed = random_forest_clf.named_steps["preprocessor"].transform(X_test)

# Convert transformed data to DataFrames for easier manipulation
X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=random_forest_clf.named_steps["preprocessor"].get_feature_names_out(),
)

X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=random_forest_clf.named_steps["preprocessor"].get_feature_names_out(),
)

# 1. Family size as a new feature
X_train_processed_df["family_size"] = X_train_processed_df["num__sibsp"] + X_train_processed_df["num__parch"]
X_test_processed_df["family_size"] = X_test_processed_df["num__sibsp"] + X_test_processed_df["num__parch"]

# 2. Fare per person (fare divided by family size + 1 to avoid division by zero)
X_train_processed_df["fare_per_person"] = X_train_processed_df["num__fare"] / (X_train_processed_df["family_size"] + 1)
X_test_processed_df["fare_per_person"] = X_test_processed_df["num__fare"] / (X_test_processed_df["family_size"] + 1)

# 3. Age group - Binning age into categories
age_bins = [0, 12, 18, 30, 50, 80]
age_labels = ["Child", "Teen", "Young Adult", "Adult", "Senior"]
X_train_processed_df["age_group"] = pd.cut(X_train_processed_df["num__age"], bins=age_bins, labels=age_labels)
X_test_processed_df["age_group"] = pd.cut(X_test_processed_df["num__age"], bins=age_bins, labels=age_labels)

# Apply one-hot encoding to age group
X_train_processed_df = pd.get_dummies(X_train_processed_df, columns=["age_group"], drop_first=True)
X_test_processed_df = pd.get_dummies(X_test_processed_df, columns=["age_group"], drop_first=True)

# Access feature importances from the RandomForestClassifier
importances = random_forest_clf.named_steps["classifier"].feature_importances_

# Ensure we get the correct feature names from the preprocessor
feature_names = random_forest_clf.named_steps["preprocessor"].get_feature_names_out()

# Check that the lengths match
if len(importances) != len(feature_names):
    print(f"Length mismatch: {len(importances)} importances vs {len(feature_names)} features")

# Create a DataFrame with feature names and their importances
feature_importances = pd.DataFrame({"feature": feature_names, "importance": importances})

# Sort features by importance
feature_importances = feature_importances.sort_values(by="importance", ascending=False)

# Display top features
print("Top features based on importance:")
print(feature_importances.head(10))

# Keep only the top N features (e.g., top 10 features)
top_features = feature_importances.head(10)["feature"].values
X_train_top = X_train_processed_df[top_features]
X_test_top = X_test_processed_df[top_features]

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}

# Define the scoring dictionary with multiple metrics
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score),
    "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),  # Updated for compatibility
}

# Initialize the GridSearchCV with the scoring parameter as a dictionary
grid_search = GridSearchCV(
    estimator=LGBMClassifier(random_state=42, verbose=-1),
    param_grid=param_grid,
    scoring=scoring,  # Multiple metrics
    refit="accuracy",  # Select the best model based on accuracy
    cv=5,
    n_jobs=-1,
)

# Fit the model on the training data
grid_search.fit(X_train_top, y_train)

# Display the best parameters and score for each metric
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy Score:", grid_search.cv_results_["mean_test_accuracy"][grid_search.best_index_])
print("Best F1 Score:", grid_search.cv_results_["mean_test_f1"][grid_search.best_index_])
print("Best AUC Score:", grid_search.cv_results_["mean_test_roc_auc"][grid_search.best_index_])

lgbm_clf = grid_search.best_estimator_

# Make predictions on the test set
y_pred = lgbm_clf.predict(X_test_top)
y_pred_proba = lgbm_clf.predict_proba(X_test_top)[:, 1]  # For AUC score

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("LGBMClassifier Accuracy:", accuracy)
print("LGBMClassifier F1 Score:", f1)
print("LGBMClassifier AUC Score:", roc_auc)
