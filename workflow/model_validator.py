from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ModelValidator:
    def validate(self, model, X_test, y_test, user_ids):
        """
        Validate model and generate detailed report
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Generate classification report
            report = classification_report(y_test, y_pred)

            # Create prediction DataFrame
            predictions_df = pd.DataFrame({
                'user_id': user_ids,
                'true_label': y_test,
                'predicted_label': y_pred,
                'prediction_probability': y_prob
            })

            # Log results
            logger.info("\nClassification Report:")
            logger.info(report)

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("\nFeature Importance:")
            logger.info(feature_importance)

            return predictions_df, feature_importance

        except Exception as e:
            logger.error(f"Error in model validation: {e}")
            raise

            
            return predictions_df, feature_importance
            
        except Exception as e:
            logger.error(f"Error in model validation: {e}")
            raise
