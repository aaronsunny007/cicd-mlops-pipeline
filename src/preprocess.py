import pandas as pd
import numpy as np
import os

np.random.seed(42)

print("Script started...")

data = pd.DataFrame({
    "stage_name": np.random.randint(0,3,500),
    "branch": np.random.randint(0,2,500),
    "environment": np.random.randint(0,2,500),
    "hour_of_day": np.random.randint(0,24,500),
    "duration_seconds": np.random.randint(30,500,500),
    "retry_count": np.random.randint(0,3,500),
    "failure": np.random.randint(0,2,500)
})

os.makedirs("data", exist_ok=True)

data.to_csv("data/cicd_logs.csv", index=False)

print("Dataset created successfully!")