import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/cicd_logs.csv")

X = data.drop("failure", axis=1)
y = data["failure"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow experiment
mlflow.set_tracking_uri("file:./mlruns")
with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    mlflow.sklearn.log_model(model, "model")


joblib.dump(model, "models/model.pkl")

print("Training complete!")