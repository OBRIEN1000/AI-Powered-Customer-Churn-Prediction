import os
import sys
import argparse
import joblib
import pandas as pd

from src.data_manager import load_raw_data, clean_data, split_and_save_data
from src.model_trainer import train_and_select_model, save_model_artifacts
from src.evaluator import evaluate_model

def compute_baselines(df):
    """
    Computes median for numerical and mode for categorical columns
    to serve as a 'neutral customer baseline' for local explanations.
    """
    baselines = {}
    for col in df.columns:
        if col in ["customerID", "Churn"]:
            continue
        # Check type
        if pd.api.types.is_numeric_dtype(df[col]):
            baselines[col] = float(df[col].median())
        else:
            baselines[col] = str(df[col].mode().iloc[0])
    return pd.DataFrame([baselines])

def run_training():
    print("=========================================")
    print("Starting Machine Learning Training Pipeline")
    print("=========================================")
    
    raw_data_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    processed_dir = "data/processed"
    models_dir = "models"
    
    # 1. Load data
    print(f"Loading raw data from {raw_data_path}...")
    df_raw = load_raw_data(raw_data_path)
    
    # 2. Clean data
    print("Cleaning raw data...")
    df_cleaned = clean_data(df_raw)
    
    # 3. Compute baseline customer profile from cleaned training data
    # (Before split to have full dataset profile, or post-split. Post-split is theoretically cleaner)
    # Let's do it post-split for mathematical rigor.
    
    # 4. Split and save data
    print("Splitting data into train/test sets...")
    train_path, test_path = split_and_save_data(df_cleaned, processed_dir)
    
    # Load splits
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Compute baselines on train split only (preventing test leak)
    print("Computing baseline customer profile from training set...")
    baseline_df = compute_baselines(train_df)
    
    # 5. Train models and select best
    best_name, best_pipeline = train_and_select_model(train_df)
    
    # 6. Save model and metadata (including training baselines)
    print("Saving model artifact...")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "best_model.joblib")
    
    metadata = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "baseline_df": baseline_df
    }
    joblib.dump(metadata, model_path)
    print(f"Model and metadata saved to {model_path}")
    
    # 7. Evaluate model & optimize threshold
    print("Running model diagnostics and threshold optimizer...")
    metrics = evaluate_model(model_path, test_df, output_dir="reports")
    
    print("\nTraining Pipeline Complete!")
    print(f"Best Model: {best_name}")
    print(f"Test ROC-AUC: {metrics['ROC-AUC']:.4f}")
    print(f"Test F1-Score (Optimal Threshold): {metrics['F1_Optimal']:.4f}")
    print(f"Optimal Threshold: {metrics['Optimal_Threshold']:.2f}")
    print(f"Max Business Savings: ${metrics['Max_Savings']:.2f}")

def run_server(host="127.0.0.1", port=8000):
    print("=========================================")
    print("Starting FastAPI Customer Churn Web App")
    print("=========================================")
    import uvicorn
    uvicorn.run("api.app:app", host=host, port=port, reload=True)

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Customer Churn Prediction CLI")
    parser.add_argument(
        "command",
        choices=["train", "serve", "run-all"],
        help="Command to run: 'train' (run model training), 'serve' (run prediction API), or 'run-all' (run train then serve)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="API server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API server port (default: 8000)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        run_training()
    elif args.command == "serve":
        run_server(args.host, args.port)
    elif args.command == "run-all":
        run_training()
        run_server(args.host, args.port)

if __name__ == "__main__":
    main()
