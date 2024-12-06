import click
import logging
from workflow.data_processor import DataProcessor
from workflow.feature_engineer import FeatureEngineer
from workflow.model_trainer import ModelTrainer
from workflow.model_validator import ModelValidator
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Tmall Customer Analysis CLI"""
    pass

@cli.command()
def train():
    """Train the customer analysis model"""
    try:
        # Initialize components
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer()
        
        # Load and process training data
        user_info, user_log = data_processor.load_data('train')
        user_info = data_processor.preprocess_user_info(user_info)
        user_features = data_processor.create_user_features(user_log)
        
        # Engineer features
        X, y, user_ids = feature_engineer.engineer_features(user_info, user_features)
        
        # Train model
        model = model_trainer.train(X, y)
        
        # Save model
        joblib.dump(model, 'models/customer_model.pkl')
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

@cli.command()
def validate():
    """Validate the trained model"""
    try:
        # Initialize components
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        model_validator = ModelValidator()
        
        # Load model
        model = joblib.load('models/customer_model.pkl')
        
        # Load and process test data
        user_info, user_log = data_processor.load_data('test')
        user_info = data_processor.preprocess_user_info(user_info)
        user_features = data_processor.create_user_features(user_log)
        
        # Engineer features
        X, y, user_ids = feature_engineer.engineer_features(user_info, user_features)
        
        # Validate model
        predictions, importance = model_validator.validate(model, X, y, user_ids)
        
        # Save results
        predictions.to_csv('results/predictions.csv', index=False)
        importance.to_csv('results/feature_importance.csv', index=False)
        
        logger.info("Model validation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in validation pipeline: {e}")
        raise

if __name__ == '__main__':
    cli()