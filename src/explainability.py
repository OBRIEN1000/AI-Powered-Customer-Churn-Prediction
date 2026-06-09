import pandas as pd
import numpy as np

def explain_prediction(pipeline, customer_df, baseline_df):
    """
    Computes local feature contributions for an individual customer prediction.
    It measures the marginal shift in churn probability when changing a single
    feature from the baseline (median/mode) value to the customer's actual value.
    
    Parameters:
    - pipeline: The trained scikit-learn Pipeline.
    - customer_df: pd.DataFrame with 1 row (the customer to explain).
    - baseline_df: pd.DataFrame with 1 row (the training baseline: median/mode).
    
    Returns:
    - explanation_list: List of dicts, sorted by feature impact.
      E.g., [{'feature': 'Contract', 'value': 'Month-to-month', 'impact': +0.22, 'direction': 'risk'}]
    """
    # Verify input forms
    if len(customer_df) != 1 or len(baseline_df) != 1:
        raise ValueError("customer_df and baseline_df must contain exactly 1 row.")
        
    customer_row = customer_df.iloc[0]
    baseline_row = baseline_df.iloc[0]
    
    # Calculate base probabilities
    p_baseline = float(pipeline.predict_proba(baseline_df)[:, 1][0])
    p_customer = float(pipeline.predict_proba(customer_df)[:, 1][0])
    
    explainable_features = [
        "tenure", "Contract", "MonthlyCharges", "TotalCharges", "InternetService",
        "TechSupport", "OnlineSecurity", "PaymentMethod", "PaperlessBilling",
        "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies",
        "MultipleLines", "PhoneService", "SeniorCitizen", "Partner", "Dependents",
        "gender"
    ]
    
    contributions = []
    
    for feature in explainable_features:
        if feature not in customer_df.columns:
            continue
            
        cust_val = customer_row[feature]
        base_val = baseline_row[feature]
        
        # If customer feature is same as baseline, contribution is 0
        if cust_val == base_val:
            contributions.append({
                "feature": feature,
                "value": str(cust_val),
                "impact": 0.0,
                "direction": "neutral",
                "readable": f"{feature} is normal ({cust_val})"
            })
            continue
            
        # Create perturbed profile where only 'feature' is changed to customer's value
        perturbed_df = baseline_df.copy()
        perturbed_df[feature] = cust_val
        
        # Predict probability for perturbed row
        p_perturbed = float(pipeline.predict_proba(perturbed_df)[:, 1][0])
        impact = p_perturbed - p_baseline
        
        # Determine direction and human readable description
        direction = "risk" if impact > 0.005 else ("retention" if impact < -0.005 else "neutral")
        
        readable = f"{feature}: {cust_val} vs baseline {base_val}"
        if feature == "Contract":
            if cust_val == "Month-to-month":
                readable = "Month-to-month contract increases churn risk"
            elif cust_val in ["One year", "Two year"]:
                readable = f"{cust_val} contract increases customer retention"
        elif feature == "tenure":
            readable = f"Tenure of {cust_val} months " + ("increases risk" if cust_val < 12 else "increases retention")
        elif feature == "InternetService":
            if cust_val == "Fiber optic":
                readable = "Fiber Optic internet is associated with high churn"
            elif cust_val == "No":
                readable = "No internet service reduces churn risk"
        elif feature in ["TechSupport", "OnlineSecurity"]:
            if cust_val == "No":
                readable = f"No {feature} increases churn risk"
            elif cust_val == "Yes":
                readable = f"Subscribed to {feature} increases retention"
        elif feature == "PaymentMethod":
            if cust_val == "Electronic check":
                readable = "Paying by Electronic Check increases churn risk"
            elif "automatic" in str(cust_val).lower():
                readable = f"Automatic payment ({cust_val.split(' ')[0]}) increases retention"
        elif feature == "MonthlyCharges":
            diff = cust_val - base_val
            readable = f"Monthly Charges of ${cust_val:.2f} are higher than average" if diff > 0 else f"Monthly Charges of ${cust_val:.2f} are lower than average"
            
        contributions.append({
            "feature": feature,
            "value": str(cust_val),
            "impact": float(impact),
            "direction": direction,
            "readable": readable
        })
        
    # Sort contributions by absolute impact
    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)
    return contributions

if __name__ == "__main__":
    # Test stub
    pass
