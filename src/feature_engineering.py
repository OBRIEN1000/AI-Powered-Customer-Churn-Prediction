import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for customer churn feature engineering.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. Tenure Cohorts
        if "tenure" in X.columns:
            X["TenureCohort"] = pd.cut(
                X["tenure"],
                bins=[-1, 12, 24, 48, 60, 100],
                labels=["0-12m", "12-24m", "24-48m", "48-60m", "60m+"]
            ).astype(str)
            
        # 2. Monthly to Total Charges Ratio
        if "MonthlyCharges" in X.columns and "TotalCharges" in X.columns:
            # Avoid division by zero
            X["MonthlyToTotalChargesRatio"] = X["MonthlyCharges"] / (X["TotalCharges"] + 1e-5)
            # Clip between 0 and 1 to handle potential anomalies
            X["MonthlyToTotalChargesRatio"] = X["MonthlyToTotalChargesRatio"].clip(0.0, 1.0)
            
        # 3. Total Services Count
        service_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        
        # Check how many service columns are present in the incoming data
        present_service_cols = [col for col in service_cols if col in X.columns]
        if present_service_cols:
            # Sum up 'Yes' values across these columns
            # Note: MultipleLines has 'Yes', 'No', 'No phone service'.
            # OnlineSecurity has 'Yes', 'No', 'No internet service'.
            services_sum = np.zeros(len(X))
            for col in present_service_cols:
                services_sum += (X[col] == "Yes").astype(int)
            X["TotalServicesCount"] = services_sum
        else:
            X["TotalServicesCount"] = 0.0
            
        # 4. Has Fiber Optic flag
        if "InternetService" in X.columns:
            X["HasFiberOptic"] = (X["InternetService"] == "Fiber optic").astype(int)
        else:
            X["HasFiberOptic"] = 0
            
        # 5. Combined Demographics: Partner & Dependents
        if "Partner" in X.columns and "Dependents" in X.columns:
            def get_demographic(row):
                p = row["Partner"] == "Yes"
                d = row["Dependents"] == "Yes"
                if p and d:
                    return "Partner_and_Dependents"
                elif p:
                    return "Partner_Only"
                elif d:
                    return "Dependents_Only"
                else:
                    return "Single"
            X["DemographicSegment"] = X.apply(get_demographic, axis=1)
        else:
            X["DemographicSegment"] = "Single"
            
        return X

if __name__ == "__main__":
    # Test feature engineering on a small sample dataframe
    sample_df = pd.DataFrame({
        "tenure": [5, 25, 70],
        "MonthlyCharges": [29.85, 80.0, 110.0],
        "TotalCharges": [29.85, 2000.0, 7700.0],
        "PhoneService": ["Yes", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "Yes"],
        "OnlineBackup": ["Yes", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "Yes"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "Yes"],
        "StreamingMovies": ["No", "No", "Yes"],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"]
    })
    
    fe = ChurnFeatureEngineer()
    engineered_df = fe.transform(sample_df)
    print(engineered_df[["TenureCohort", "MonthlyToTotalChargesRatio", "TotalServicesCount", "HasFiberOptic", "DemographicSegment"]])
