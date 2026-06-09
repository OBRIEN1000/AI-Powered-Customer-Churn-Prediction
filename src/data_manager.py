import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw_data(filepath):
    """Loads the raw customer churn dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Cleans the raw dataset:
    - Trims whitespace from string columns.
    - Handles TotalCharges empty strings by converting to 0.0 (since tenure is 0 for these customers).
    - Maps target Churn column 'Yes'/'No' to 1/0.
    """
    df = df.copy()
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        
    # TotalCharges contains empty strings ' ' for new customers with tenure = 0.
    # We replace them with '0.0' and cast to float.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace("", "0.0")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        
    # Map target Churn to binary 1/0
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
        
    return df

def split_and_save_data(df, output_dir, test_size=0.2, random_state=42):
    """
    Performs stratified split and saves train and test sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Stratified split based on target Churn
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Churn"] if "Churn" in df.columns else None,
        random_state=random_state
    )
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Data split complete: Train size = {len(train_df)}, Test size = {len(test_df)}")
    print(f"Saved to {train_path} and {test_path}")
    return train_path, test_path

if __name__ == "__main__":
    # Test script locally
    raw_path = "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    processed_dir = "../data/processed"
    if os.path.exists(raw_path):
        data = load_raw_data(raw_path)
        cleaned = clean_data(data)
        split_and_save_data(cleaned, processed_dir)
