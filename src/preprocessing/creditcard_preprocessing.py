# src/preprocessing/creditcard_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os
import logging
import yaml
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom log transformer"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)
    
    def get_feature_names_out(self, input_features=None):
        return [f"log_{name}" for name in input_features]

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def validate_data(df):
    """Validate input data structure and values"""
    required_columns = ['Time', 'Amount', 'Class']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Ensure Class is binary (0 or 1)
    if not df['Class'].isin([0, 1]).all():
        invalid_classes = df['Class'].unique()
        raise ValueError(f"Target column 'Class' contains invalid values: {invalid_classes}")
    
    return True

def create_preprocessing_pipeline():
    """Create preprocessing pipeline excluding Class column"""
    # Create transformers
    time_amt_scaler = RobustScaler()
    log_transformer = LogTransformer()
    log_scaler = RobustScaler()
    
    # Create column transformer (EXCLUDE Class from transformation)
    preprocessor = ColumnTransformer(
        transformers=[
            ('time_amt', time_amt_scaler, ['Time', 'Amount']),
            ('log_amt', Pipeline([
                ('log', log_transformer), 
                ('scale', log_scaler)
            ]), ['Amount'])
        ],
        remainder='passthrough'
    )
    return preprocessor

def preprocess_creditcard(input_path, output_path, config):
    """Preprocess credit card transaction data while preserving Class"""
    try:
        logger.info("Loading credit card data...")
        cc_data = pd.read_csv(input_path)
        
        logger.info(f"Original dataset size: {len(cc_data)} records")
        logger.info(f"Columns: {list(cc_data.columns)}")
        
        # Data validation
        validate_data(cc_data)
        
        # Handle missing values
        initial_missing = cc_data.isna().sum().sum()
        if initial_missing > 0:
            logger.warning(f"Missing values found: {initial_missing}")
            cc_data = cc_data.dropna()
        
        # Remove duplicates
        duplicates = cc_data.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate records")
            cc_data = cc_data.drop_duplicates()
        
        # Preserve Class column before transformation
        class_data = cc_data['Class'].copy()
        
        # Create features
        logger.info("Creating financial risk features...")
        cc_data['Hour'] = cc_data['Time'] % 24
        cc_data['Transaction_Value_Ratio'] = cc_data['Amount'] / cc_data['Amount'].mean()
        
        # Preprocessing pipeline (EXCLUDE Class)
        features = cc_data.drop(columns=['Class'])
        preprocessor = create_preprocessing_pipeline()
        processed_data = preprocessor.fit_transform(features)
        
        # Get feature names
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'log_amt':
                feature_names.extend([f"log_scaled_{col}" for col in cols])
            elif name == 'time_amt':
                feature_names.extend([f"scaled_{col}" for col in cols])
            elif name == 'remainder':
                # Exclude Class from remainder features
                remainder_cols = features.columns.difference(['Time', 'Amount', 'Class'])
                feature_names.extend(remainder_cols)
        
        # Create processed DataFrame and add Class back
        cc_data_processed = pd.DataFrame(processed_data, columns=feature_names)
        cc_data_processed['Class'] = class_data.values
        
        # Save processor
        os.makedirs(config.get('models_dir', './models'), exist_ok=True)
        joblib.dump(preprocessor, os.path.join(config.get('models_dir', './models'), 'creditcard_preprocessor.pkl'))
        
        # Save processed data
        cc_data_processed.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        logger.info(f"Final dataset size: {len(cc_data_processed)} records")
        logger.info(f"Class distribution: \n{cc_data_processed['Class'].value_counts()}")
        
        return cc_data_processed
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    config = load_config()
    input_path = config.get('data', {}).get('creditcard_raw', './data/raw/creditcard.csv')
    output_path = config.get('data', {}).get('processed_creditcard', './data/processed/processed_creditcard.csv')
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    preprocess_creditcard(input_path, output_path, config)