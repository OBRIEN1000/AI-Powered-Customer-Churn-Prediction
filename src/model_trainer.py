import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from src.feature_engineering import ChurnFeatureEngineer
from src.preprocessor import get_preprocessor

def build_pipeline(classifier):
    """
    Assembles the complete ML pipeline:
    1. Custom feature engineering
    2. Column preprocessing (scaling/encoding)
    3. Classifier
    """
    return Pipeline(steps=[
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocessor", get_preprocessor()),
        ("classifier", classifier)
    ])

def train_and_select_model(train_df, random_state=42):
    """
    Compares multiple models using GridSearchCV and returns the best model.
    """
    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    
    # Define candidate models and hyperparameter grids
    models_config = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=random_state, class_weight="balanced"),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [5, 8, 12],
                "classifier__min_samples_split": [2, 5]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "params": {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.01, 0.1]
            }
        }
    }
    
    best_score = 0.0
    best_name = None
    best_pipeline = None
    
    print("Starting model selection via 5-fold Cross-Validation (optimizing ROC-AUC)...")
    
    for name, config in models_config.items():
        print(f"\nTuning {name}...")
        pipeline = build_pipeline(config["model"])
        
        # Grid Search with 5-fold cross validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid=config["params"],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        mean_cv_score = grid_search.best_score_
        print(f"{name} Best CV ROC-AUC: {mean_cv_score:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_name = name
            best_pipeline = grid_search.best_estimator_
            
    print(f"\nBest Model: {best_name} with CV ROC-AUC = {best_score:.4f}")
    
    # Fit the best pipeline on the entire training set
    print("Refitting best model on full training set...")
    best_pipeline.fit(X_train, y_train)
    
    return best_name, best_pipeline

def save_model_artifacts(model_pipeline, model_name, output_dir="models"):
    """Saves the trained pipeline to disk."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "best_model.joblib")
    
    # Save model alongside its training metadata
    metadata = {
        "model_name": model_name,
        "pipeline": model_pipeline
    }
    
    joblib.dump(metadata, filepath)
    print(f"Model and metadata successfully saved to {filepath}")
    return filepath

if __name__ == "__main__":
    # Local quick test
    train_path = "../data/processed/train.csv"
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        name, pipe = train_and_select_model(train_df)
        save_model_artifacts(pipe, name, "../models")
