from kfp.dsl import (
    pipeline,
    Input,
    Output,
    Artifact,
    container_component,
    ContainerSpec,
)
from kfp import compiler


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
        image="asia-south1-docker.pkg.dev/vertex-ai-im/mlops-demo/restaurant-nn:latest",
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
    input: Input[Artifact],
    out_data: Output[Artifact],
):
    return ContainerSpec(
        image="asia-south1-docker.pkg.dev/vertex-ai-im/mlops-demo/restaurant-nn:latest",
        command=["python3", "data-validation.py"],
        args=[
            "--mlflow_uri",
            mlflow_uri,
            "--input",
            input_file,
        ],
    )


@container_component
def data_preprocessing(
    input_file: str,
    model_folder: str,
    property_type: str,
    batch_size: int,
    out_data: Output[Artifact],
):
    return ContainerSpec(
        image="asia-south1-docker.pkg.dev/vertex-ai-im/mlops-demo/restaurant-nn:latest",
        command=["python3", "batch_predict.py"],
        args=[
            "--input-file",
            input_file,
            "--model-folder",
            model_folder,
            "--property-type",
            property_type,
            "--batch-size",
            batch_size,
            "--output-folder",
            out_data.path,
        ],
    )
