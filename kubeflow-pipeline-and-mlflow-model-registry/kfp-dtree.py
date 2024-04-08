from kfp.dsl import (
    pipeline,
    Input,
    Output,
    Artifact,
    Model,
    Metrics,
    container_component,
    ContainerSpec,
)
from kfp import compiler

# TODO: int this pipeline we are using csv file as data source
# @container_component
# def data_extraction():
#     pass


@container_component
def data_preprocessing(
    mlflow_uri: str,
    input_file: str,
    out_data: Output[Artifact],
):
    return ContainerSpec(
        image="sumitkushwah/restaurant-dtree:latest",
        command=["python3", "data-pre-processing.py"],
        args=[
            "--mlflow_uri",
            mlflow_uri,
            "--input",
            input_file,
            "--output",
            out_data.path,
        ],
    )


@container_component
def data_validation(
    mlflow_uri: str,
    output: Output[Metrics],
    input: Input[Artifact],
):
    return ContainerSpec(
        image="sumitkushwah/restaurant-dtree:latest",
        command=["python3", "data-validation.py"],
        args=[
            "--mlflow_uri",
            mlflow_uri,
            "--input",
            input.path,
        ],
    )


@container_component
def data_preparation(
    mlflow_uri: str,
    not_used: Input[Metrics],
    input: Input[Artifact],
    output: Output[Artifact],
):
    return ContainerSpec(
        image="sumitkushwah/restaurant-dtree:latest",
        command=["python3", "data_preparation.py"],
        args=[
            "--mlflow_uri",
            mlflow_uri,
            "--input",
            input.path,
            "--output",
            output.path,
        ],
    )


@container_component
def model_train(
    mlflow_uri: str,
    input: Input[Artifact],
    model: Output[Model],
):
    return ContainerSpec(
        image="sumitkushwah/restaurant-dtree:latest",
        command=["python3", "nn_train.py"],
        args=[
            "--mlflow_uri",
            mlflow_uri,
            "--x-train-path",
            input.path,
            "--y-train-path",
            input.path,
            "--model-out-dir",
            model.path,
        ],
    )


@container_component
def model_eval(
    mlflow_uri: str,
    input: Input[Artifact],
    model: Input[Model],
):
    return ContainerSpec(
        image="sumitkushwah/restaurant-dtree:latest",
        command=["python3", "model_evaluation.py"],
        args=[
            "--mlflow_uri",
            mlflow_uri,
            "--x-test-path",
            input.path,
            "--y-test-path",
            input.path,
            "--model-file-path",
            model.path,
        ],
    )


# BUCKET_URI = "gs://vertex-ai-im"
# PIPELINE_ROOT = "{}/pipeline-runs".format(BUCKET_URI)


@pipeline(name="restaurant-model-pipeline")
def restaurant_pipeline(
    input_file: str = "/data/raw-data-restaurant.csv",
    mlflow_uri: str = "http://68.183.90.6:32699/",
):
    data_preprocessing_task = data_preprocessing(
        mlflow_uri=mlflow_uri,
        input_file=input_file,
    )
    data_validation_task = data_validation(
        mlflow_uri=mlflow_uri,
        input=data_preprocessing_task.outputs["out_data"],
    )
    data_preparation_task = data_preparation(
        mlflow_uri=mlflow_uri,
        not_used=data_validation_task.outputs["output"],
        input=data_preprocessing_task.outputs["out_data"],
    )
    model_train_task = model_train(
        mlflow_uri=mlflow_uri,
        input=data_preparation_task.outputs["output"],
    )
    model_eval(
        mlflow_uri=mlflow_uri,
        input=data_preparation_task.outputs["output"],
        model=model_train_task.outputs["model"],
    )


compiler.Compiler().compile(
    pipeline_func=restaurant_pipeline, package_path="res_pipeline_dtree.yaml"
)
