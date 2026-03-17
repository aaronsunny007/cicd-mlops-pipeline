# CI/CD MLOps Pipeline Project

This project demonstrates a complete **Machine Learning Operations (MLOps) pipeline** for predicting failures in CI/CD pipelines.

## Project Overview

The system trains a machine learning model to predict whether a CI/CD pipeline stage will fail based on execution features such as:

* stage name
* branch
* environment
* execution duration
* retry count
* hour of execution

The model is then served through a **FastAPI prediction API** and packaged using **Docker**.

---

## Architecture

```
Synthetic Data Generator
        ↓
Data Preprocessing
        ↓
Model Training (Scikit-learn)
        ↓
MLflow Experiment Tracking
        ↓
Model Artifact (model.pkl)
        ↓
FastAPI Inference API
        ↓
Docker Container
```

---

## Tech Stack

* Python
* Scikit-learn
* FastAPI
* MLflow
* Docker
* GitHub

---

## Running the Project

### Train the model

```
python src/train.py
```

### Start the API

```
uvicorn api.app:app --reload
```

### Run with Docker

```
docker build -t cicd-mlops .
docker run -p 8000:8000 cicd-mlops
```

API available at:

```
http://localhost:8000/docs
```

---

## Example Prediction Request

```
POST /predict
```

```json
{
 "stage_name": 1,
 "branch": 0,
 "environment": 1,
 "hour_of_day": 14,
 "duration_seconds": 200,
 "retry_count": 1
}
```

---

## Future Improvements

* CI/CD automation with GitHub Actions
* Model monitoring
* Cloud deployment
* Kubernetes scaling
