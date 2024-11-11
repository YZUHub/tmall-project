import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):
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
    grid_search.fit(X_train, y_train)

    # Display the best parameters and score for each metric
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy Score:", grid_search.cv_results_["mean_test_accuracy"][grid_search.best_index_])
    print("Best F1 Score:", grid_search.cv_results_["mean_test_f1"][grid_search.best_index_])
    print("Best AUC Score:", grid_search.cv_results_["mean_test_roc_auc"][grid_search.best_index_])

    classifier = grid_search.best_estimator_
    joblib.dump(classifier, "models/classifier.pkl")
    print("Trained etl model saved as 'models/classifier.pkl'")
