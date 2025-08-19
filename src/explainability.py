# explainability.py
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")


def generate_shap_plots(
    model, X_test, feature_names, output_dir, dataset_name, model_type
):
    """Generate SHAP explainability plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Create explainer
    if model_type == "logistic":
        explainer = shap.LinearExplainer(model, X_test)
    else:  # XGBoost
        explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary - {dataset_name} {model_type}")
    summary_path = os.path.join(output_dir, f"{dataset_name}_{model_type}_summary.png")
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.title(f"SHAP Feature Importance - {dataset_name} {model_type}")
    bar_path = os.path.join(output_dir, f"{dataset_name}_{model_type}_bar.png")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()

    # Force plot for a specific example
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0, :],
        X_test.iloc[0, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.title(f"SHAP Force Plot - {dataset_name} {model_type}")
    force_path = os.path.join(output_dir, f"{dataset_name}_{model_type}_force.png")
    plt.savefig(force_path, bbox_inches="tight")
    plt.close()

    return summary_path, bar_path, force_path


def run_explainability():
    """Run SHAP explainability for all models"""
    datasets = ["ecommerce", "creditcard"]
    model_types = ["logistic", "xgboost"]
    output_dir = "./results/shap_plots"

    results = []

    for dataset in datasets:
        for model_type in model_types:
            print(f"\nGenerating SHAP plots for {dataset} {model_type}...")

            # Load model
            model_path = f"./models/{dataset}/{model_type}_model.pkl"
            model = joblib.load(model_path)

            # Load test data
            test_path = f"./data/balanced/{dataset}_test.csv"
            test_data = pd.read_csv(test_path)
            target_col = "class" if dataset == "ecommerce" else "Class"
            X_test = test_data.drop(columns=[target_col])

            # Generate plots
            feature_names = X_test.columns.tolist()
            summary, bar, force = generate_shap_plots(
                model, X_test, feature_names, output_dir, dataset, model_type
            )

            results.append(
                {
                    "dataset": dataset,
                    "model": model_type,
                    "summary_plot": summary,
                    "bar_plot": bar,
                    "force_plot": force,
                }
            )

    print("\nSHAP analysis completed!")
    return results


if __name__ == "__main__":
    run_explainability()
