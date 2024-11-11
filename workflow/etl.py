import joblib
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data():
    # Load the Titanic dataset from Seaborn
    data = sns.load_dataset("titanic")

    # Drop rows with missing values in the target column ('survived') for simplicity
    data = data.dropna(subset=["survived"])

    # Define target and features
    X = data.drop("survived", axis=1)
    y = data["survived"]
    return X, y


def split_data(X, y):
    # Split into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def configure_preprocessor_pipeline():
    # Define numerical and categorical feature groups
    numerical_features = ["age", "sibsp", "parch", "fare"]
    categorical_features = ["pclass", "sex", "embarked", "who", "adult_male", "alone"]
    drop_features = ["deck", "embark_town", "class", "alive"]

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
    return preprocessor


def train_grids(algorithm, model_params, grid_params, X_train, X_test, y_train, y_test):
    # Full pipeline with preprocessor and model
    model_pipeline = Pipeline(
        [
            ("preprocessor", configure_preprocessor_pipeline()),
            ("classifier", algorithm(**model_params)),  # Substitute with any other classifier as needed
        ]
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


def process_data():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Split dataset for transformation using preprocessor_pipeline
    X_train_imp, X_test_imp, y_train_imp, y_test_imp = split_data(X_train, y_train)

    # Define the algorithm and model parameters
    grid_params = {
        "n_estimators": [20, 30, 40, 50, 60],  # Number of trees in the forest
        "max_depth": [3, 4, 5, 6],  # Maximum depth of the tree
        "max_features": [3, 5, 10],
    }
    model_params = {"random_state": 42}

    classifier = train_grids(
        RandomForestClassifier,
        model_params,
        grid_params,
        X_train_imp,
        X_test_imp,
        y_train_imp,
        y_test_imp,
    )
    joblib.dump(classifier, "models/etl-model.pkl")
    print("Trained etl model saved as 'models/etl-model.pkl'")

    # Transform X_train using the fitted pipeline
    X_train_processed = classifier.named_steps["preprocessor"].transform(X_train)
    # Convert transformed data to DataFrames for easier manipulation
    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=classifier.named_steps["preprocessor"].get_feature_names_out(),
    )

    return X_train_processed_df, X_test, y_train, y_test
