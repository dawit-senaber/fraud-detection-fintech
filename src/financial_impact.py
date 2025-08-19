def calculate_savings(y_true, y_pred, amounts, fp_cost=10, fn_cost=100):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    savings = (fn * amounts[y_true == 1].mean() * fn_cost) - (fp * fp_cost)
    return {
        "false_positives": fp,
        "false_negatives": fn,
        "savings": savings,
        "roi": savings / (fp * fp_cost + 1e-6),
    }
