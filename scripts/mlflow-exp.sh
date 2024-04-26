export MLFLOW_URI=http://127.0.0.1:5000/
export EXP_NAME="Restaurant-Turnover"
export DIR="open-source-mlops/kubeflow-pipeline-and-mlflow-model-registry"
export MODEL_TYPE="nn"
python $DIR/data-pre-processing.py --input temp-data/raw-data-restaurant.csv --output temp-data/ --mlflow_uri $MLFLOW_URI --mlflow_exp $EXP_NAME

python $DIR/data-validation.py --input temp-data/ --mlflow_uri $MLFLOW_URI --mlflow_exp $EXP_NAME

python $DIR/$MODEL_TYPE/data_preparation.py --input temp-data/ --mlflow_uri $MLFLOW_URI --output temp-data/ --mlflow_exp $EXP_NAME

python $DIR/$MODEL_TYPE/nn_train.py --x-train-path temp-data/ --y-train-path temp-data/ --mlflow_uri $MLFLOW_URI --model-out-dir temp-data/ --mlflow_exp $EXP_NAME

python $DIR/$MODEL_TYPE/model_evaluation.py --x-test-path temp-data/ --y-test-path temp-data/ --mlflow_uri $MLFLOW_URI --model-file-path temp-data/ --mlflow_exp $EXP_NAME