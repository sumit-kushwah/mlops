mlflow models build-docker \
    --model-uri "runs:/36cb4104e40449cf8252cad19c01a5b5/model" \
    --name "mlflow-serving-res-turnover"

mlflow models generate-dockerfile \
    --model-uri "runs:/36cb4104e40449cf8252cad19c01a5b5/model" \
    -d model-serving/with-mlflow/mlflow-generate-dockerfile