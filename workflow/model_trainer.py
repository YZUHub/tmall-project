from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }

    def train(self, X_train, y_train):
        """
        Train the model using GridSearchCV with cross-validation.
        Args:
            X_train (array-like): Features for training.
            y_train (array-like): Labels for training.

        Returns:
            best_estimator_: The best trained model.
        """
        try:
            logger.info("Starting model training with GridSearchCV...")
            
            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=5,  # 5-fold cross-validation
                n_jobs=-1,  # Use all available cores
                scoring='f1',  # Scoring metric
            )
            
            # Fit the grid search to the training data
            grid_search.fit(X_train, y_train)
            
            # Log results
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Return the best model
            return grid_search.best_estimator_
        
        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            raise ve
        except Exception as e:
            logger.error(f"Unexpected error during model training: {e}")
            raise e
