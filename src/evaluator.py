import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix
)

def evaluate_model(model_path, test_df, c_offer=20.0, l_churn=150.0, output_dir="reports"):
    """
    Evaluates the model on test data, runs threshold optimization,
    generates performance reports, and saves plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load model metadata
    metadata = joblib.load(model_path)
    pipeline = metadata["pipeline"]
    model_name = metadata["model_name"]
    
    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]
    
    # 2. Get predictions
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # 3. Cost-Sensitive Threshold Optimization
    thresholds = np.linspace(0, 1, 101)
    savings = []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        # confusion_matrix structure:
        # [[TN, FP],
        #  [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        
        # Financial utility calculation:
        # Without model, cost is: (tp + fn) * l_churn (all actual churners leave)
        # With model, cost is: (tp + fp) * c_offer (intervention offers) + fn * l_churn (missed churners leave)
        # Savings = Cost(no model) - Cost(model)
        #         = (tp + fn)*l_churn - (tp + fp)*c_offer - fn*l_churn
        #         = tp * (l_churn - c_offer) - fp * c_offer
        net_savings = tp * (l_churn - c_offer) - fp * c_offer
        savings.append(net_savings)
        
    best_idx = np.argmax(savings)
    optimal_threshold = thresholds[best_idx]
    max_savings = savings[best_idx]
    
    print(f"\n--- Business Optimization ({model_name}) ---")
    print(f"Assumed Campaign Cost per Customer: ${c_offer}")
    print(f"Assumed Lost Value per Churner: ${l_churn}")
    print(f"Optimal Probability Threshold: {optimal_threshold:.2f}")
    print(f"Maximized Financial Savings: ${max_savings:,.2f} (on test set)")
    
    # Calculate metrics at default (0.5) and optimal thresholds
    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Accuracy_Default": accuracy_score(y_test, y_pred_default),
        "Precision_Default": precision_score(y_test, y_pred_default),
        "Recall_Default": recall_score(y_test, y_pred_default),
        "F1_Default": f1_score(y_test, y_pred_default),
        "Accuracy_Optimal": accuracy_score(y_test, y_pred_optimal),
        "Precision_Optimal": precision_score(y_test, y_pred_optimal),
        "Recall_Optimal": recall_score(y_test, y_pred_optimal),
        "F1_Optimal": f1_score(y_test, y_pred_optimal),
        "Optimal_Threshold": optimal_threshold,
        "Max_Savings": max_savings
    }
    
    # Update model artifact with threshold parameters and metrics
    metadata["optimal_threshold"] = float(optimal_threshold)
    metadata["c_offer"] = float(c_offer)
    metadata["l_churn"] = float(l_churn)
    metadata["metrics"] = metrics
    
    # Extract Feature Importances
    feature_importances = get_feature_importances(pipeline)
    metadata["feature_importances"] = feature_importances
    
    joblib.dump(metadata, model_path)
    print(f"Updated model metadata in {model_path} with optimal threshold and metrics.")
    
    # 4. Generate Diagnostic Plots
    plot_evaluation_curves(y_test, y_prob, thresholds, savings, optimal_threshold, max_savings, output_dir)
    plot_confusion_matrices(y_test, y_pred_default, y_pred_optimal, optimal_threshold, output_dir)
    plot_feature_importances(feature_importances, output_dir)
    
    return metrics

def get_feature_importances(pipeline):
    """
    Extracts feature importances or coefficients from the pipeline.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    # Attempt to get feature names after one-hot encoding
    try:
        feature_names = preprocessor.get_feature_names_out()
        # Clean up feature name prefixes (e.g. 'cat__Contract_One year' -> 'Contract_One year')
        feature_names = [name.split("__")[-1] for name in feature_names]
    except Exception:
        # Fallback names
        feature_names = [f"Feature {i}" for i in range(100)]
        
    # Get importances or coefficients
    importances = None
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        
    if importances is not None:
        # Align importances with feature names
        num_features = min(len(feature_names), len(importances))
        feature_names = feature_names[:num_features]
        importances = importances[:num_features]
        
        # Sort
        sorted_idx = np.argsort(importances)[::-1]
        importance_list = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in sorted_idx
        ]
        return importance_list
    return []

def plot_evaluation_curves(y_test, y_prob, thresholds, savings, optimal_threshold, max_savings, output_dir):
    """Generates and saves ROC, PR, and Cost-savings curves."""
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].legend(loc="lower right")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recall, precision, color="blue", lw=2, label="Precision-Recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")
    
    # Cost-Savings Curve
    axes[2].plot(thresholds, savings, color="green", lw=2, label="Net Savings ($)")
    axes[2].axvline(optimal_threshold, color="red", linestyle="--", label=f"Optimal threshold ({optimal_threshold:.2f})")
    axes[2].scatter([optimal_threshold], [max_savings], color="red", zorder=5)
    axes[2].set_xlabel("Probability Threshold")
    axes[2].set_ylabel("Savings ($)")
    axes[2].set_title("Business Net Savings vs. Threshold")
    axes[2].legend(loc="lower center")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_performance_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Performance curves plot saved to {plot_path}")

def plot_confusion_matrices(y_test, y_pred_default, y_pred_optimal, optimal_threshold, output_dir):
    """Plots confusion matrices comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_default = confusion_matrix(y_test, y_pred_default)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    
    sns.heatmap(cm_default, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Confusion Matrix (Threshold = 0.50)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_xticklabels(["Retain", "Churn"])
    axes[0].set_yticklabels(["Retain", "Churn"])
    
    sns.heatmap(cm_optimal, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False)
    axes[1].set_title(f"Confusion Matrix (Optimized Threshold = {optimal_threshold:.2f})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_xticklabels(["Retain", "Churn"])
    axes[1].set_yticklabels(["Retain", "Churn"])
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Confusion matrix comparisons saved to {plot_path}")

def plot_feature_importances(feature_importances, output_dir, top_n=15):
    """Plots the top N feature importances."""
    if not feature_importances:
        return
        
    df_fi = pd.DataFrame(feature_importances[:top_n])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=df_fi, hue="feature", legend=False, palette="viridis")
    plt.title(f"Top {top_n} Global Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "feature_importances.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")

if __name__ == "__main__":
    # Local quick test
    test_path = "../data/processed/test.csv"
    model_path = "../models/best_model.joblib"
    if os.path.exists(test_path) and os.path.exists(model_path):
        test_df = pd.read_csv(test_path)
        metrics = evaluate_model(model_path, test_df)
        print(metrics)
