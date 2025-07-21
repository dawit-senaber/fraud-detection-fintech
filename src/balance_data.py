# balance_data.py
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def print_class_distribution(y, title):
    counter = Counter(y)
    print(f"\n{title} class distribution:")
    for class_val, count in counter.items():
        percent = 100 * count / len(y)
        print(f"Class {class_val}: {count} ({percent:.2f}%)")

def prepare_features(df, target_col, dataset_name):
    """Prepare features by dropping non-numerical columns and encoding categorical variables"""
    # Drop identifier and datetime columns
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'ip_address', 'device_id']
    if dataset_name == 'ecommerce':
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical features for ecommerce
    if dataset_name == 'ecommerce':
        cat_cols = ['source', 'browser', 'sex', 'country']
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_cols))
        
        # Combine with numerical features
        num_cols = [col for col in X.columns if col not in cat_cols]
        X = pd.concat([X[num_cols].reset_index(drop=True), 
                      encoded_df.reset_index(drop=True)], axis=1)
    
    return X, y

def balance_dataset(dataset_path, target_col, output_dir, dataset_name):
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name} dataset")
    print(f"Loading data from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    print(f"Original dataset size: {len(df)} records")
    print_class_distribution(df[target_col], "Original")
    
    # Prepare features
    X, y = prepare_features(df, target_col, dataset_name)
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print_class_distribution(y_train, "Training (before balancing)")
    print_class_distribution(y_test, "Testing")
    
    # Apply SMOTE-ENN only on training data
    print("\nApplying SMOTE-ENN to training data...")
    smote_enn = SMOTEENN(random_state=42)
    X_res, y_res = smote_enn.fit_resample(X_train, y_train)
    
    print_class_distribution(y_res, "Training (after balancing)")
    
    # Create balanced DataFrames
    train_balanced = pd.DataFrame(X_res, columns=X.columns)
    train_balanced[target_col] = y_res
    
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df[target_col] = y_test
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    train_path = f"{output_dir}/{dataset_name}_train_balanced.csv"
    test_path = f"{output_dir}/{dataset_name}_test.csv"
    
    train_balanced.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved balanced training data to: {train_path}")
    print(f"Saved test data to: {test_path}")
    print(f"Balanced training size: {len(train_balanced)}")
    print(f"Test size: {len(test_df)}")
    
    return train_balanced, test_df

if __name__ == "__main__":
    # E-commerce data
    balance_dataset(
        './data/processed_ecommerce.csv',
        'class',
        './data/balanced',
        'ecommerce'
    )
    
    # Credit card data
    balance_dataset(
        './data/processed_creditcard.csv',
        'Class',
        './data/balanced',
        'creditcard'
    )