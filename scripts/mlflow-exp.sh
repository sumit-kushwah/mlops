export MLFLOW_URI=http://127.0.0.1:5000/
python data-pre-processing.py --input ../restaurant-annual-turnover-model-exp/dataset/raw-data-restaurant.csv --output ../restaurant-annual-turnover-model-exp/dataset/ --mlflow_uri $MLFLOW_URI

python data-validation.py --input ../restaurant-annual-turnover-model-exp/dataset/ --mlflow_uri $MLFLOW_URI

python dtree/data_preparation.py --input ../restaurant-annual-turnover-model-exp/dataset/ --mlflow_uri $MLFLOW_URI --output ../restaurant-annual-turnover-model-exp/dataset/

python dtree/dtree_train.py --x-train-path ../restaurant-annual-turnover-model-exp/dataset/ --y-train-path ../restaurant-annual-turnover-model-exp/dataset/ --mlflow_uri $MLFLOW_URI --model-out-dir ../restaurant-annual-turnover-model-exp/dataset/

python dtree/model_evaluation.py --x-test-path ../restaurant-annual-turnover-model-exp/dataset/ --y-test-path ../restaurant-annual-turnover-model-exp/dataset/ --mlflow_uri $MLFLOW_URI --model-file-path ../restaurant-annual-turnover-model-exp/dataset/