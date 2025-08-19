def calculate_financial_impact(y_true, y_pred, amounts, fp_cost=10, fn_cost=100):
    """
    Calculate financial impact of fraud decisions
    
    y_true: Actual labels (1 = fraud)
    y_pred: Predicted labels (1 = fraud)
    amounts: Transaction amounts
    fp_cost: Cost of false positive ($)
    fn_cost: Cost of false negative ($)
    """
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    
    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()
    
    fp_loss = fp_count * fp_cost
    fn_loss = fn_mask.dot(amounts) * fn_cost
    
    total_loss = fp_loss + fn_loss
    savings = fn_mask.dot(amounts) - total_loss
    
    return {
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'total_loss': total_loss,
        'potential_savings': savings,
        'roi': savings / (total_loss + 1e-6)
    }