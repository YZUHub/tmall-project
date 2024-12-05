import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def engineer_features(self, user_info, user_features):
        """
        Combine and engineer features from user info and behavior
        """
        try:
            # Merge user info with behavior features
            final_features = pd.merge(user_info, user_features, on='user_id', how='left')
            
            # Create feature columns
            feature_cols = ['age_range', 'gender', 'total_interactions', 'unique_items',
                          'unique_categories', 'unique_merchants', 'unique_brands',
                          'purchase_count', 'purchase_ratio']
            
            X = final_features[feature_cols]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
            
            # Create target variable (example: high-value customer based on purchase_count)
            y = (final_features['purchase_count'] > final_features['purchase_count'].median()).astype(int)
            
            return X_scaled, y, final_features['user_id']
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise