# ChurnShield: AI-Powered Customer Churn Analytics Platform

ChurnShield is a production-grade, end-to-end machine learning system designed to predict customer churn, optimize financial intervention campaigns, and serve predictions in real time. The project combines modular ML pipelines, RESTful APIs, and an interactive business dashboard for decision makers.

---

## 🌟 Key Features

1. **Modular Data & ML Pipelines**: Standardized codebase separating data validation, preprocessing, feature engineering, and cross-validated model selection.
2. **Business-Cost Threshold Optimization**: Moving away from the default $0.50$ probability threshold, the model uses a financial utility model to find the classification cutoff that maximizes campaign savings, balancing retention costs ($20) against churn lifetime value lost ($150).
3. **Local Explainability Engine**: A perturbation-based attribution model that calculates feature-level contributions (drivers) for any specific customer, explaining predictions without complex C-library compile dependencies.
4. **FastAPI Prediction Service**: REST endpoints for single-customer inference, drag-and-drop batch CSV uploads, and asynchronous background model retraining.
5. **Interactive What-If Dashboard**: A premium dark-mode SPA (HTML/CSS/JS) featuring real-time risk gauges, what-if sliders, local explanation bars, and live pipeline retraining logs.

---

## 📁 Project Architecture

```
├── api/
│   ├── app.py                # FastAPI web server & prediction routes
│   └── static/               # Web client static assets
│       ├── index.html        # SPA dashboard structure
│       ├── style.css         # Custom dark-theme styling
│       └── app.js            # Reactive bindings & Chart.js integrations
├── data/
│   ├── raw/                  # Raw Telco CSV database
│   └── processed/            # Cleaned splits (train.csv, test.csv)
├── models/
│   └── best_model.joblib     # Serialized classifier & preprocessor pipeline
├── reports/                  # Evaluator charts (ROC, PR, Cost curves)
├── src/                      # Core machine learning package
│   ├── data_manager.py       # Data cleaning & stratified splits
│   ├── evaluator.py          # Metrics scoring & business cost optimizer
│   ├── explainability.py     # Perturbation-based local explainability
│   ├── feature_engineering.py  # Domain feature custom transformers
│   ├── model_trainer.py      # Grid search comparisons
│   └── preprocessor.py       # ColumnTransformer scaling & encoding
├── main.py                   # Unified CLI runner
├── Procfile                  # Process manager config for cloud hosts
└── requirements.txt          # Python dependencies
```

---

## Getting Started

### 1. Setup Environment
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Ingest, Train, and Evaluate
Run the end-to-end training pipeline to ingest the database, execute parameter search, compute business thresholds, and export diagnostic curves under `reports/`:
```bash
python main.py train
```

### 3. Run the Web Server
Launch the FastAPI prediction service and serve the dashboard locally:
```bash
python main.py serve
```
Open **[http://localhost:8000/](http://localhost:8000/)** in your browser to access the dashboard.

---

## ☁️ Cloud Deployment

ChurnShield is designed for instant containerized or server-based cloud deployments:

### Deploying to Render or Railway (Recommended)
1. Push your repository to **GitHub**.
2. Create a new **Web Service** on Render or Railway.
3. Link your GitHub repository.
4. Set the following parameters:
   - **Environment / Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt && python main.py train`
   - **Start Command**: `uvicorn api.app:app --host 0.0.0.0 --port $PORT` (this is automatically picked up from the `Procfile`).
5. Render will automatically build the dataset, train the best model, save the artifacts, and start the FastAPI service with a public SSL URL!


check it on render : https://ai-powered-customer-churn-prediction.onrender.com/
