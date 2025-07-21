# creditcard_preprocessing.py
import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def preprocess_creditcard(input_path, output_path):
    # Load data
    print("Loading credit card data...")
    cc_data = pd.read_csv(input_path)
    
    print(f"Original dataset size: {len(cc_data)} records")
    
    # Check for missing values
    missing_values = cc_data.isna().sum().sum()
    print(f"Missing values before cleaning: {missing_values}")
    
    # Drop duplicates
    duplicates = cc_data.duplicated().sum()
    print(f"Duplicate records found: {duplicates}")
    cc_data = cc_data.drop_duplicates()
    
    # Handle missing values
    cc_data = cc_data.dropna()
    
    # Scale 'Time' and 'Amount'
    print("Scaling features...")
    scaler = RobustScaler()
    cc_data[['Time', 'Amount']] = scaler.fit_transform(cc_data[['Time', 'Amount']])
    
    # Check class distribution
    fraud_count = cc_data['Class'].sum()
    fraud_rate = fraud_count / len(cc_data)
    print(f"Fraud count: {fraud_count}")
    print(f"Fraud rate: {fraud_rate:.6f}")
    
    # Save processed data
    cc_data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Final dataset size: {len(cc_data)} records")
    
    return cc_data

if __name__ == "__main__":
    input_path = './data/creditcard.csv'
    output_path = './data/processed_creditcard.csv'
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    preprocess_creditcard(input_path, output_path)