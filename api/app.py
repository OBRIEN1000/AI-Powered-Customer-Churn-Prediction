import os
import joblib
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.explainability import explain_prediction
from src.data_manager import clean_data

app = FastAPI(
    title="AI-Powered Customer Churn Prediction Service",
    description="REST API for predicting customer churn risk, providing local model explanations, and optimizing retention actions.",
    version="1.0.0"
)

# Global variables to hold model and metadata
MODEL_PATH = "models/best_model.joblib"
model_metadata = None
pipeline = None
optimal_threshold = 0.5
baseline_df = None

def load_model():
    global model_metadata, pipeline, optimal_threshold, baseline_df
    if os.path.exists(MODEL_PATH):
        try:
            model_metadata = joblib.load(MODEL_PATH)
            pipeline = model_metadata["pipeline"]
            optimal_threshold = model_metadata.get("optimal_threshold", 0.5)
            baseline_df = model_metadata.get("baseline_df", None)
            print(f"Model successfully loaded. Best model: {model_metadata.get('model_name', 'Unknown')}, Optimal Threshold: {optimal_threshold:.2f}")
        except Exception as e:
            print(f"Error loading model: {e}")
            pipeline = None
    else:
        print(f"Model file not found at {MODEL_PATH}. Prediction endpoints will be disabled until trained.")

# Initial load
load_model()

# Define customer schemas
class CustomerData(BaseModel):
    customerID: Optional[str] = "7590-VHVEG"
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., description="Number of months customer has stayed")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)")
    MonthlyCharges: float = Field(..., description="Monthly amount charged to customer")
    TotalCharges: float = Field(..., description="Total amount charged to customer (send float directly)")

class SinglePredictResponse(BaseModel):
    customerID: str
    churn_probability: float
    is_high_risk: bool
    optimal_threshold: float
    recommended_action: str
    explanations: List[dict]

def generate_retention_recommendation(customer_dict, explanations):
    """
    Generates personalized business-oriented recommendations based on 
    the customer's profile and top risk factors.
    """
    contract = customer_dict.get("Contract", "")
    internet = customer_dict.get("InternetService", "")
    tech_support = customer_dict.get("TechSupport", "")
    payment = customer_dict.get("PaymentMethod", "")
    monthly_charges = customer_dict.get("MonthlyCharges", 0.0)
    tenure = customer_dict.get("tenure", 0)
    
    # Check top negative features (those with direction == 'risk')
    risk_features = [exp["feature"] for exp in explanations if exp["direction"] == "risk"]
    
    if not risk_features:
        return "No action required. Customer is highly loyal and satisfied."
        
    # Order of recommendations based on business priority
    if contract == "Month-to-month":
        discount = 15 if monthly_charges > 70 else 10
        return (f"Offer transition to a 1-Year Contract with a loyalty discount of ${discount}/month. "
                "Moving from month-to-month contracts is our highest leverage retention mechanism.")
                
    if internet == "Fiber optic" and tech_support == "No":
        return ("Provide a complimentary 3-month trial of Tech Support & Online Security. "
                "Fiber optic customers have high bandwidth but exhibit high churn when support services are missing.")
                
    if payment == "Electronic check":
        return ("Promote automatic payment methods (Credit Card or Bank Transfer) by offering a "
                "one-time billing credit of $10. Auto-pay customers have significantly higher retention rates.")
                
    if monthly_charges > 85 and tenure < 18:
        return ("Recommend transitioning to a family package or standard plan with slightly lower rates, "
                "or offer a high-value value-add service for free (e.g. Device Protection) to offset price sensitivity.")
                
    return "Schedule a standard proactive customer success check-in call to review service satisfaction."

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "model_name": model_metadata.get("model_name", "None") if model_metadata else "None",
        "optimal_threshold": optimal_threshold
    }

@app.get("/api/metrics")
def get_metrics():
    if not model_metadata:
        raise HTTPException(status_code=503, detail="Model is not available. Please run training first.")
        
    return {
        "model_name": model_metadata["model_name"],
        "optimal_threshold": optimal_threshold,
        "c_offer": model_metadata.get("c_offer", 20.0),
        "l_churn": model_metadata.get("l_churn", 150.0),
        "metrics": model_metadata.get("metrics", {}),
        "feature_importances": model_metadata.get("feature_importances", [])[:15]
    }

@app.post("/api/predict", response_model=SinglePredictResponse)
def predict_single(customer: CustomerData):
    if not pipeline or baseline_df is None:
        raise HTTPException(status_code=503, detail="Model or training baseline is not loaded.")
        
    # Convert input to DataFrame
    customer_dict = customer.model_dump()
    customer_id = customer_dict.pop("customerID", "Unknown")
    
    # Model pipeline expects exact pandas DF structure
    df_customer = pd.DataFrame([customer_dict])
    
    # Calculate churn probability
    try:
        y_prob = float(pipeline.predict_proba(df_customer)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    # Decide based on optimized threshold
    is_high_risk = y_prob >= optimal_threshold
    
    # Generate local explanations using our custom explainability engine
    explanations = explain_prediction(pipeline, df_customer, baseline_df)
    
    # Generate action plan
    recommendation = generate_retention_recommendation(customer_dict, explanations)
    
    return SinglePredictResponse(
        customerID=customer_id,
        churn_probability=y_prob,
        is_high_risk=is_high_risk,
        optimal_threshold=optimal_threshold,
        recommended_action=recommendation,
        explanations=explanations
    )

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if not pipeline or baseline_df is None:
        raise HTTPException(status_code=503, detail="Model or training baseline is not loaded.")
        
    # Check file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")
        
    try:
        # Load CSV
        df = pd.read_csv(file.file)
        
        # Keep customerID
        customer_ids = df["customerID"] if "customerID" in df.columns else [f"Cust_{i}" for i in range(len(df))]
        
        # Clean the input dataframe just like the training data
        df_cleaned = clean_data(df)
        
        # Remove target Churn if present, and remove customerID
        X_batch = df_cleaned.drop(columns=["customerID", "Churn"], errors="ignore")
        
        # Predict
        probs = pipeline.predict_proba(X_batch)[:, 1]
        
        # Build response rows
        results = []
        for idx, prob in enumerate(probs):
            prob_val = float(prob)
            is_risk = prob_val >= optimal_threshold
            
            # Extract row dict for recommendation generator
            cust_dict = X_batch.iloc[idx].to_dict()
            
            # Simple local explanation for recommendation (top features)
            # To save computation time for large batches, we only run full explainability
            # for high-risk customers or provide a lighter version.
            # Here we run a fast explanations call:
            df_cust_row = pd.DataFrame([cust_dict])
            explanations = explain_prediction(pipeline, df_cust_row, baseline_df)
            
            recommendation = generate_retention_recommendation(cust_dict, explanations)
            
            results.append({
                "customerID": str(customer_ids.iloc[idx]),
                "churn_probability": round(prob_val, 4),
                "churn_decision": "High Risk" if is_risk else "Low Risk",
                "recommended_action": recommendation
            })
            
        return {"total_customers": len(results), "predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

def background_retrain_task():
    try:
        from main import run_training
        print("Background retraining task started...")
        run_training()
        load_model()
        print("Background retraining task successfully completed and model reloaded.")
    except Exception as e:
        print(f"Error during background retraining: {e}")

@app.post("/api/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_retrain_task)
    return {"status": "retraining_initiated", "message": "Model retraining is running in the background."}

# Mount static and reports folders
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(reports_dir, exist_ok=True)

# Serves index.html at '/'
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(
        content="<h2>AI-Powered Customer Churn Dashboard</h2><p>Static index.html not found yet. Please wait for front-end assets generation.</p>",
        status_code=404
    )

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")
