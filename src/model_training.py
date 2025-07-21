# model_training.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, f1_score, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Predict probabilities and classes
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'auc_pr': average_precision_score(y_test, y_pred_proba),
        'f1_score': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics

def train_and_evaluate(dataset_name, model_type, train_path, test_path, target_col):
    """Train and evaluate a model"""
    print(f"\n{'='*50}")
    print(f"Training {model_type} model on {dataset_name} data...")
    
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]
    
    # Initialize model
    if model_type == 'logistic':
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear',
            random_state=42
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
    else:
        raise ValueError("Invalid model type")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    model_dir = f'./models/{dataset_name}'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_type}_model.pkl')
    joblib.dump(model, model_path)
    
    print(f"Model saved to: {model_path}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return {
        'dataset': dataset_name,
        'model_type': model_type,
        'metrics': metrics,
        'model_path': model_path
    }

if __name__ == "__main__":
    datasets = {
        'ecommerce': {
            'train_path': './data/balanced/ecommerce_train_balanced.csv',
            'test_path': './data/balanced/ecommerce_test.csv',
            'target_col': 'class'
        },
        'creditcard': {
            'train_path': './data/balanced/creditcard_train_balanced.csv',
            'test_path': './data/balanced/creditcard_test.csv',
            'target_col': 'Class'
        }
    }
    
    results = []
    
    for dataset_name, data_info in datasets.items():
        for model_type in ['logistic', 'xgboost']:
            result = train_and_evaluate(
                dataset_name=dataset_name,
                model_type=model_type,
                train_path=data_info['train_path'],
                test_path=data_info['test_path'],
                target_col=data_info['target_col']
            )
            results.append(result)
    
    # Save results
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    results_df = pd.DataFrame([{
        'dataset': r['dataset'],
        'model': r['model_type'],
        'auc_roc': r['metrics']['auc_roc'],
        'auc_pr': r['metrics']['auc_pr'],
        'f1_score': r['metrics']['f1_score'],
        'model_path': r['model_path']
    } for r in results])
    
    results_path = os.path.join(results_dir, 'model_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print("\nTraining completed!")
    print(f"Results saved to: {results_path}")
    print("\nSummary of results:")
    print(results_df[['dataset', 'model', 'auc_roc', 'auc_pr', 'f1_score']])