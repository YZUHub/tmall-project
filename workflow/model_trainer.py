from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    
    def train(self, X_train, y_train):
        """
        Train model with grid search CV
        """
        try:
            logger.info("Starting model training with grid search...")
            
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                cv=5,
                n_jobs=-1,
                scoring='f1'
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise