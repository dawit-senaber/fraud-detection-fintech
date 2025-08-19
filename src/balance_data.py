# src/balance_data.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import Counter
import logging
import yaml
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}  # Return empty dict if config fails

def print_class_distribution(y, title):
    """Print class distribution with enhanced logging"""
    counter = Counter(y)
    logger.info(f"\n{title} class distribution:")
    for class_val, count in counter.items():
        percent = 100 * count / len(y)
        logger.info(f"Class {class_val}: {count} ({percent:.2f}%)")

def prepare_features(df, target_col, dataset_name):
    """Prepare features without encoding - just drop unnecessary columns"""
    # Ensure target is integer type for classification
    df[target_col] = df[target_col].astype(int)
    
    # Drop identifier and datetime columns
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'ip_address', 'device_id']
    if dataset_name == 'ecommerce':
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def handle_imbalance(X, y, min_samples=5):
    """Ensure minimum samples per class for balancing"""
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()
    
    if len(valid_classes) < 2:
        raise ValueError(f"Only {len(valid_classes)} valid classes found. Need at least 2 classes with min {min_samples} samples each.")
    
    # Filter data to include only valid classes
    valid_indices = y.isin(valid_classes)
    return X[valid_indices], y[valid_indices]

def finance_specific_balancing(X_train, y_train, risk_ratio=10):
    """Custom balancing strategy for financial fraud detection"""
    # Get class distribution
    class_counts = y_train.value_counts()
    min_class = class_counts.idxmin()
    maj_class = class_counts.idxmax()
    
    # Calculate desired ratio (1:risk_ratio for fraud detection)
    n_minority = class_counts[min_class]
    n_majority = min(n_minority * risk_ratio, class_counts[maj_class])
    
    # Downsample majority class
    majority_indices = y_train[y_train == maj_class].sample(n_majority, random_state=42).index
    minority_indices = y_train[y_train == min_class].index
    
    # Combine indices
    balanced_indices = majority_indices.union(minority_indices)
    
    return X_train.loc[balanced_indices], y_train.loc[balanced_indices]

def balance_dataset(dataset_path, target_col, output_dir, dataset_name, config):
    """Balance dataset with finance-specific handling"""
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {dataset_name} dataset")
        logger.info(f"Loading data from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        logger.info(f"Original dataset size: {len(df)} records")
        print_class_distribution(df[target_col], "Original")
        
        # Prepare features
        X, y = prepare_features(df, target_col, dataset_name)
        
        # Ensure minimum samples per class
        X, y = handle_imbalance(X, y, min_samples=5)
        
        # Train-test split (stratified if possible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )
            logger.info("Using stratified sampling")
        except ValueError:
            logger.warning("Stratified sampling failed. Using random sampling")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        print_class_distribution(y_train, "Training (before balancing)")
        print_class_distribution(y_test, "Testing")
        
        # Apply finance-specific balancing
        logger.info("Applying finance-specific balancing...")
        
        # Get risk ratio from config or use default
        risk_ratio = config.get('balancing', {}).get('risk_ratio', 10)
        logger.info(f"Using risk ratio: 1:{risk_ratio}")
        
        X_res, y_res = finance_specific_balancing(X_train, y_train, risk_ratio)
        
        print_class_distribution(y_res, "Training (after balancing)")
        
        # Create balanced DataFrames
        train_balanced = pd.DataFrame(X_res)
        train_balanced[target_col] = y_res
        
        test_df = pd.DataFrame(X_test)
        test_df[target_col] = y_test
        
        # Save datasets
        os.makedirs(output_dir, exist_ok=True)
        train_path = f"{output_dir}/{dataset_name}_train_balanced.csv"
        test_path = f"{output_dir}/{dataset_name}_test.csv"
        
        train_balanced.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"\nSaved balanced training data to: {train_path}")
        logger.info(f"Saved test data to: {test_path}")
        logger.info(f"Balanced training size: {len(train_balanced)}")
        logger.info(f"Test size: {len(test_df)}")
        
        return train_balanced, test_df
        
    except Exception as e:
        logger.error(f"Balancing failed for {dataset_name}: {str(e)}")
        raise

if __name__ == "__main__":
    config = load_config()
    
    # E-commerce data
    balance_dataset(
        config.get('data', {}).get('processed_ecommerce', './data/processed_ecommerce.csv'),
        'class',
        config.get('data', {}).get('balanced_dir', './data/balanced'),
        'ecommerce',
        config
    )
    
    # Credit card data
    balance_dataset(
        config.get('data', {}).get('processed_creditcard', './data/processed_creditcard.csv'),
        'Class',
        config.get('data', {}).get('balanced_dir', './data/balanced'),
        'creditcard',
        config
    )