import os
import mlflow

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "/home/sumit/Downloads/vertex-ai-im-0dab9dd74d97.json"
)

mlflow.set_tracking_uri("http://35.244.48.239:8000/")


model_info = mlflow.models.get_model_info(
    "runs:/36cb4104e40449cf8252cad19c01a5b5/model"
)

print(model_info.artifact_path)

import mlflow.pyfunc

model_name = "Restaurant-Turnover"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
