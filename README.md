# SmartPrice Predictor – Production ML Inference Service

An end-to-end machine learning system for predicting product prices from unstructured text and numeric features.  
The project covers the full lifecycle from experimentation and model comparison to model registry, latency benchmarking, and cloud deployment.

---

## Production Endpoint

**Live API:**  
[https://smartprice-predictor.onrender.com/docs](https://smartprice-api-ru6k.onrender.com/docs)

---

## Dataset

Dataset used for training and evaluation:  
https://www.kaggle.com/datasets/raghavdharwal/amazon-ml-challenge-2025

---

## MLflow Experiments

All experiments, metrics, and model versions are tracked in MLflow:  
https://dagshub.com/reddyrohith49471/amazon-ml-2025/experiments

---

## Problem Statement

The objective of this project is to predict product prices using:

- Unstructured product text (combined_text)
- A numeric feature (Value)

The target variable is processed using:

- Outlier clipping
- Log transformation (price_log)

The goal is to build a model that achieves strong predictive performance while maintaining low inference latency for production deployment.

---

## What Was Done

The project followed a structured experimentation and deployment workflow:

- Built baseline models using TF-IDF features with linear and ridge regression.
- Evaluated nonlinear models including gradient boosting.
- Tested transformer-based text embeddings combined with classical models.
- Compared all models using RMSE and SMAPE metrics.
- Selected the best performing pipeline: TF-IDF + XGBoost.
- Logged all experiments and metrics using MLflow.
- Registered the best model in the MLflow Model Registry.
- Benchmarked inference latency using the production model.
- Converted the notebook workflow into a modular, production-style codebase.
- Built a FastAPI service that loads the model directly from the registry.
- Deployed the service as a cloud-hosted inference API.

---

## Model Experiment Results

| Model | RMSE | SMAPE |
|------|------|------|
| Linear Regression + TF-IDF | ~35.16 | ~59.20% |
| Ridge Regression + TF-IDF | ~35.09–36.41 | ~57–59% |
| HistGradientBoosting | ~34.98 | ~56.40% |
| Transformer + Ridge | ~35.5+ | Higher error |
| Transformer + XGBoost | ~37+ | Higher error |
| TF-IDF + XGBoost (Final Model) | ~34.70 | ~56.18% |

Final model selected based on lowest RMSE and stable SMAPE.

---

## Experiment Tracking

All experiments were:

- Logged using MLflow
- Compared using RMSE and SMAPE
- Versioned through the MLflow Model Registry
- Promoted to a Production environment for deployment


---

## Latency Results

### Notebook Inference Latency

Measured directly from the loaded production model:

| Metric | Latency |
|--------|---------|
| Average | ~3.47 ms |
| Median | ~3.38 ms |
| P95 | ~3.79 ms |
| Max | ~6.98 ms |

---

### API Latency (Local Deployment)

Measured through the FastAPI service:

- Average latency: ~3-6 ms per request

This includes:

- Request parsing
- Prediction
- Response serialization

---

## Architecture

The system is organized into modular, production-oriented components.

```
Client Request
      │
      ▼
FastAPI Inference Layer
      │
      ▼
Prediction Service
      │
      ▼
MLflow Model Loader
      │
      ▼
Registered Production Model
```

---

## Local Setup

### 1. Clone the repository
```
git clone https://github.com/reddyrohith49471/SmartPrice_Predictor.git
cd smartprice-predictor
```

### 2. Create virtual environment
```
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Run the service
```
python -m uvicorn src.api.app:app
```

Open in browser:
```
http://127.0.0.1:8000/docs
```

---

## Deployment

The service is configured for cloud deployment using:

```
render.yaml
```

Start command:
```
python -m uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
```

After deployment, the model is served through a public inference endpoint.

---

## Future Improvements

### Modeling
- Incorporate product image features
- Build multimodal models (text + image + numeric)
- Automated hyperparameter tuning

### MLOps
- Data drift detection
- Prediction drift monitoring
- Automated retraining pipelines
- Scheduled evaluation jobs

### System Enhancements
- Batch inference endpoints
- Model version switching via API
- Caching for repeated predictions
- Containerized deployment with Docker


