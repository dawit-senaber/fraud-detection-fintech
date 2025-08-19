#!/usr/bin/env python3
"""
FRAUDGUARD PRO - Complete Processing Pipeline
Author: Dawit Senaber
Date: August 19, 2025

This script runs the complete fraud detection pipeline:
1. Data preprocessing
2. Data balancing
3. Model training
4. Explainability analysis
5. Performance reporting
"""

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

# Add the parent directory to Python path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_preprocessing():
    """Run data preprocessing for both datasets"""
    logger.info("Starting data preprocessing...")
    
    try:
        # Import preprocessing modules
        from src.preprocessing.creditcard_preprocessing import preprocess_creditcard
        from src.preprocessing.ecommerce_preprocessing import preprocess_ecommerce
        
        # Run credit card preprocessing
        logger.info("Processing credit card data...")
        preprocess_creditcard(
            './data/creditcard.csv',
            './data/processed_creditcard.csv'
        )
        
        # Run e-commerce preprocessing
        logger.info("Processing e-commerce data...")
        preprocess_ecommerce(
            './data/Fraud_Data.csv',
            './data/IpAddress_to_Country.csv',
            './data/processed_ecommerce.csv'
        )
        
        logger.info("Data preprocessing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return False

def run_balancing():
    """Run data balancing for both datasets"""
    logger.info("Starting data balancing...")
    
    try:
        # Import balancing module
        from src.balance_data import balance_dataset
        
        # Balance credit card data
        logger.info("Balancing credit card data...")
        balance_dataset(
            './data/processed_creditcard.csv',
            'Class',
            './data/balanced',
            'creditcard'
        )
        
        # Balance e-commerce data
        logger.info("Balancing e-commerce data...")
        balance_dataset(
            './data/processed_ecommerce.csv',
            'class',
            './data/balanced',
            'ecommerce'
        )
        
        logger.info("Data balancing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data balancing failed: {str(e)}")
        return False

def run_training():
    """Run model training for both datasets"""
    logger.info("Starting model training...")
    
    try:
        # Import training module
        from src.model_training import train_and_evaluate
        
        # Define dataset configurations
        datasets = {
            'creditcard': {
                'train_path': './data/balanced/creditcard_train_balanced.csv',
                'test_path': './data/balanced/creditcard_test.csv',
                'target_col': 'Class'
            },
            'ecommerce': {
                'train_path': './data/balanced/ecommerce_train_balanced.csv',
                'test_path': './data/balanced/ecommerce_test.csv',
                'target_col': 'class'
            }
        }
        
        results = []
        
        # Train models for each dataset
        for dataset_name, data_info in datasets.items():
            for model_type in ['logistic', 'xgboost']:
                logger.info(f"Training {model_type} model on {dataset_name} data...")
                
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
        
        logger.info(f"Model training completed. Results saved to: {results_path}")
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_explainability():
    """Run SHAP explainability analysis"""
    logger.info("Starting SHAP explainability analysis...")
    
    try:
        # Import explainability module
        from src.explainability import run_explainability
        
        results = run_explainability()
        
        logger.info("SHAP analysis completed successfully")
        logger.info(f"Generated {len(results)} SHAP plot sets")
        return True
        
    except Exception as e:
        logger.error(f"SHAP analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_report():
    """Generate a comprehensive performance report"""
    logger.info("Generating performance report...")
    
    try:
        # Load results
        results_path = './results/model_results.csv'
        if not os.path.exists(results_path):
            logger.warning("No results found to generate report")
            return False
            
        results_df = pd.read_csv(results_path)
        
        # Create report
        report = f"""
FRAUDGUARD PRO - PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL PERFORMANCE SUMMARY:
{results_df.to_string(index=False)}

BEST PERFORMING MODELS BY DATASET:
"""
        # Add best models for each dataset
        for dataset in ['creditcard', 'ecommerce']:
            dataset_results = results_df[results_df['dataset'] == dataset]
            if not dataset_results.empty:
                best_model = dataset_results.loc[dataset_results['auc_roc'].idxmax()]
                report += f"\n{dataset.upper()}: {best_model['model']} (AUC-ROC: {best_model['auc_roc']:.4f})"
        
        # Save report
        report_path = './results/performance_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Performance report saved to: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(description='FRAUDGUARD PRO Processing Pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing')
    parser.add_argument('--skip-balancing', action='store_true', help='Skip data balancing')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-explainability', action='store_true', help='Skip SHAP analysis')
    parser.add_argument('--skip-report', action='store_true', help='Skip report generation')
    
    args = parser.parse_args()
    
    logger.info("Starting FRAUDGUARD PRO processing pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create necessary directories
    os.makedirs('./data/balanced', exist_ok=True)
    os.makedirs('./models/creditcard', exist_ok=True)
    os.makedirs('./models/ecommerce', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Run pipeline steps
    steps = [
        ("Preprocessing", run_preprocessing, args.skip_preprocessing),
        ("Balancing", run_balancing, args.skip_balancing),
        ("Training", run_training, args.skip_training),
        ("Explainability", run_explainability, args.skip_explainability),
        ("Reporting", generate_report, args.skip_report)
    ]
    
    success = True
    for step_name, step_func, skip in steps:
        if skip:
            logger.info(f"Skipping {step_name} step")
            continue
            
        logger.info(f"Starting {step_name} step")
        step_success = step_func()
        
        if not step_success:
            logger.error(f"{step_name} step failed")
            success = False
            break
            
        logger.info(f"Completed {step_name} step")
    
    if success:
        logger.info("Pipeline completed successfully")
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("Next steps:")
        print("1. Launch dashboard: streamlit run app.py")
        print("2. View results in ./results/ directory")
        print("="*50)
    else:
        logger.error("Pipeline failed")
        print("\n" + "="*50)
        print("PIPELINE FAILED")
        print("Check pipeline.log for details")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main()