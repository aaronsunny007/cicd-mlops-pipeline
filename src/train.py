import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure directories exist
os.makedirs("mlruns", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Set MLflow tracking URI to local folder
mlflow.set_tracking_uri("file:./mlruns")

# Start MLflow experiment
mlflow.set_experiment("cicd_failure_prediction")

# Load dataset
data = pd.read_csv("data/cicd_logs.csv")

X = data.drop("failure", axis=1)
y = data["failure"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    # Log model to MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

# Save model locally
joblib.dump(model, "models/model.pkl")

print("Training complete!")