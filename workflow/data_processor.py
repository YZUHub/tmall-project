import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
from .config import DATA_FILES, FEATURES

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all data files"""
        data_frames = {}
        for name, path in DATA_FILES.items():
            try:
                data_frames[name] = pd.read_csv(path)
                print(f"Loaded {name} from {path}")
            except Exception as e:
                print(f"Error loading {name}: {str(e)}")
        return data_frames
    
    def merge_data(self, data_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all data frames"""
        # Merge logic based on common keys
        if 'user_info' in data_frames and 'user_log' in data_frames:
            df = pd.merge(
                data_frames['user_info'],
                data_frames['user_log'],
                on='user_id',
                how='left'
            )
            
            if 'user_log2' in data_frames:
                df = pd.merge(
                    df,
                    data_frames['user_log2'],
                    on='user_id',
                    how='left'
                )
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data"""
        # Handle missing values
        df = df.fillna({
            col: df[col].mean() for col in FEATURES['numerical_cols']
        })
        df = df.fillna({
            col: 'unknown' for col in FEATURES['categorical_cols']
        })
        
        # Encode categorical variables
        for col in FEATURES['categorical_cols']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        # Select features
        feature_cols = FEATURES['numerical_cols'] + FEATURES['categorical_cols']
        X = df[feature_cols]
        
        # Scale numerical features
        X_scaled = X.copy()
        X_scaled[FEATURES['numerical_cols']] = self.scaler.fit_transform(
            X[FEATURES['numerical_cols']]
        )
        
        # Prepare target if exists
        y = None
        if FEATURES['target_col'] in df.columns:
            y = df[FEATURES['target_col']]
            
        return X_scaled, y