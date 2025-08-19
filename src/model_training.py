import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import os
import logging
import yaml
import matplotlib.pyplot as plt
import shap
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects specific features from a DataFrame"""

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]


def create_preprocessor(dataset_name):
    """Create preprocessing pipeline for datasets"""
    if dataset_name == "ecommerce":
        cat_cols = ["source", "browser", "sex", "country"]
        num_cols = ["purchase_value", "time_since_signup", "purchase_hour", "age"]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                ),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop",
        )

        return Pipeline(
            [
                ("selector", FeatureSelector(cat_cols + num_cols)),
                ("preprocessor", preprocessor),
            ]
        )

    elif dataset_name == "creditcard":
        # Credit card features are already preprocessed
        return None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_model(model, X_test, y_test, dataset_name, model_type, config):
    """Evaluate model performance with finance-specific metrics"""
    # Predict probabilities and classes
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Calculate core metrics
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_pred_proba),
        "auc_pr": average_precision_score(y_test, y_pred_proba),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    # Financial impact metrics
    fp_cost = config.get("financial_impact", {}).get("false_positive_cost", 10)
    fn_cost = config.get("financial_impact", {}).get("false_negative_cost", 100)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["false_positive_cost"] = fp * fp_cost
    metrics["false_negative_cost"] = fn * fn_cost
    metrics["total_cost"] = (
        metrics["false_positive_cost"] + metrics["false_negative_cost"]
    )

    # Find optimal threshold for financial cost
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    costs = []
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        cost = fp * fp_cost + fn * fn_cost
        costs.append(cost)

    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    metrics["optimal_threshold"] = optimal_threshold
    metrics["min_cost"] = costs[optimal_idx]

    # Plot financial cost curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, "b-", label="Total Cost")
    plt.axvline(
        x=optimal_threshold, color="r", linestyle="--", label="Optimal Threshold"
    )
    plt.xlabel("Threshold")
    plt.ylabel("Financial Cost ($)")
    plt.title(f"Financial Cost Curve - {dataset_name} {model_type}")
    plt.legend()

    cost_dir = config.get("results", {}).get("cost_dir", "./results/cost_curves")
    os.makedirs(cost_dir, exist_ok=True)
    cost_plot_path = f"{cost_dir}/{dataset_name}_{model_type}_cost_curve.png"
    plt.savefig(cost_plot_path)
    plt.close()
    metrics["cost_plot"] = cost_plot_path

    return metrics


def train_xgboost(X_train, y_train):
    """Train XGBoost model with finance-optimized parameters"""
    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    # Base model
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",  # Optimize for AUC-PR (fraud detection)
        use_label_encoder=False,
        random_state=42,
    )

    # Parameter grid for tuning
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
    }

    # Use stratified k-fold for financial data
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="average_precision",  # Focus on precision-recall
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Performing grid search for XGBoost...")
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best AUC-PR: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_logistic(X_train, y_train):
    """Train logistic regression with finance-specific calibration"""
    # Base model with increased max_iter
    base_model = LogisticRegression(
        class_weight="balanced",
        max_iter=10000,  # Increased for convergence
        solver="liblinear",
        penalty="l1",  # L1 regularization for feature selection
        random_state=42,
    )

    # Only calibrate if needed - return base model for SHAP compatibility
    return base_model.fit(X_train, y_train)


def generate_feature_importance(model, X_test, dataset_name, model_type):
    """Generate SHAP feature importance analysis with proper model handling"""
    try:
        # Handle different model types
        if model_type == "logistic":
            # For logistic models, use LinearExplainer
            explainer = shap.LinearExplainer(model, X_test)
            shap_values = explainer.shap_values(X_test)
        elif model_type == "xgboost":
            # For XGBoost, use TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"Feature Importance - {dataset_name} {model_type}")

        shap_dir = "./results/shap_plots"
        os.makedirs(shap_dir, exist_ok=True)
        plot_path = f"{shap_dir}/{dataset_name}_{model_type}_feature_importance.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        return plot_path

    except Exception as e:
        logger.error(
            f"Feature importance failed for {dataset_name} {model_type}: {str(e)}"
        )
        return None


def train_and_evaluate(
    dataset_name, model_type, train_path, test_path, target_col, config
):
    """Train and evaluate a model with financial optimizations"""
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type} model on {dataset_name} data...")

        # Load data
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        # Separate features and target
        X_train = train.drop(columns=[target_col])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col])
        y_test = test[target_col]

        # Create and apply preprocessor for e-commerce
        preprocessor = None
        if dataset_name == "ecommerce":
            preprocessor = create_preprocessor(dataset_name)
            logger.info("Fitting preprocessor for e-commerce data...")
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

        # Initialize model
        if model_type == "logistic":
            model = train_logistic(X_train, y_train)
        elif model_type == "xgboost":
            model = train_xgboost(X_train, y_train)
        else:
            raise ValueError("Invalid model type")

        # Evaluate model
        metrics = evaluate_model(
            model, X_test, y_test, dataset_name, model_type, config
        )

        # Save model
        model_dir = f"{config.get('models_dir', './models')}/{dataset_name}"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
        joblib.dump(model, model_path)

        # Save preprocessor if exists
        preprocessor_path = None
        if preprocessor:
            preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved to: {preprocessor_path}")

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        logger.info(f"Minimum Financial Cost: ${metrics['min_cost']:.2f}")

        return {
            "dataset": dataset_name,
            "model_type": model_type,
            "metrics": metrics,
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
        }

    except Exception as e:
        logger.error(f"Training failed for {dataset_name} {model_type}: {str(e)}")
        raise


if __name__ == "__main__":
    config = load_config()
    datasets = {
        "ecommerce": {
            "train_path": f"{config.get('data', {}).get('balanced_dir', './data/balanced')}/ecommerce_train_balanced.csv",
            "test_path": f"{config.get('data', {}).get('balanced_dir', './data/balanced')}/ecommerce_test.csv",
            "target_col": "class",
        },
        "creditcard": {
            "train_path": f"{config.get('data', {}).get('balanced_dir', './data/balanced')}/creditcard_train_balanced.csv",
            "test_path": f"{config.get('data', {}).get('balanced_dir', './data/balanced')}/creditcard_test.csv",
            "target_col": "Class",
        },
    }

    results = []

    for dataset_name, data_info in datasets.items():
        for model_type in ["logistic", "xgboost"]:
            result = train_and_evaluate(
                dataset_name=dataset_name,
                model_type=model_type,
                train_path=data_info["train_path"],
                test_path=data_info["test_path"],
                target_col=data_info["target_col"],
                config=config,
            )
            results.append(result)

            # Generate feature importance
            model = joblib.load(result["model_path"])

            # Load test data
            test_df = pd.read_csv(data_info["test_path"])
            X_test = test_df.drop(columns=[data_info["target_col"]])

            # For e-commerce, apply preprocessing if needed
            if dataset_name == "ecommerce" and result.get("preprocessor_path"):
                preprocessor = joblib.load(result["preprocessor_path"])
                X_test = preprocessor.transform(X_test)

            importance_path = generate_feature_importance(
                model, X_test, dataset_name, model_type
            )
            if importance_path:
                result["metrics"]["feature_importance"] = importance_path

    # Save results
    results_dir = config.get("results", {}).get("base_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)
    results_df = pd.DataFrame(
        [
            {
                "dataset": r["dataset"],
                "model": r["model_type"],
                "auc_roc": r["metrics"]["auc_roc"],
                "auc_pr": r["metrics"]["auc_pr"],
                "f1_score": r["metrics"]["f1_score"],
                "optimal_threshold": r["metrics"]["optimal_threshold"],
                "min_cost": r["metrics"]["min_cost"],
                "model_path": r["model_path"],
                "preprocessor_path": r.get("preprocessor_path", ""),
                "feature_importance": r["metrics"].get("feature_importance", ""),
                "cost_plot": r["metrics"]["cost_plot"],
            }
            for r in results
        ]
    )

    results_path = os.path.join(results_dir, "model_results.csv")
    results_df.to_csv(results_path, index=False)

    logger.info("\nTraining completed!")
    logger.info(f"Results saved to: {results_path}")
    logger.info("\nSummary of results:")
    logger.info(
        results_df[
            ["dataset", "model", "auc_roc", "auc_pr", "f1_score", "min_cost"]
        ].to_string()
    )
