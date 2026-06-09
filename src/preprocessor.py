from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Define the features that will be present AFTER feature engineering
NUMERIC_FEATURES = [
    "tenure", 
    "MonthlyCharges", 
    "TotalCharges", 
    "TotalServicesCount", 
    "MonthlyToTotalChargesRatio",
    "HasFiberOptic"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureCohort",
    "DemographicSegment"
]

def get_preprocessor():
    """
    Returns a ColumnTransformer containing:
    - Standard Scaling for numeric features.
    - One-hot encoding for categorical features, with handle_unknown='ignore'
      to prevent errors on unseen data during web-serving.
    """
    # Pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combined column preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop" # Drop customerID and any other unused columns
    )
    
    return preprocessor

if __name__ == "__main__":
    # Test preprocessor structure
    prep = get_preprocessor()
    print("Preprocessor initialized successfully:")
    print(prep)
