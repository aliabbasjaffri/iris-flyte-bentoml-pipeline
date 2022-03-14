"""
Flyte Pipeline for Iris Dataset classification
------------
"""
try:
    from train import train_iris_dataset, calculate_accuracy
    from model import IrisClassificationModel
    from datasource import load_iris_dataset, scale_iris_dataset, train_test_data_split
except ImportError:
    from .train import train_iris_dataset, calculate_accuracy
    from .model import IrisClassificationModel
    from .datasource import load_iris_dataset, scale_iris_dataset, train_test_data_split
import os
import torch
import wandb
import bentoml
import pandas as pd
from numpy import ndarray
from torch.autograd import Variable
from flytekit import task, workflow


wandb.init(
    project="iris-classifier",
    entity=os.environ.get("WANDB_USERNAME", "aliabbasjaffri"),
)


@task
def get_iris_data() -> (ndarray, ndarray, ndarray, ndarray):
    data, target, target_names, feature_names = load_iris_dataset()
    return data, target, target_names, feature_names


@task
def scale_iris_data(data: ndarray) -> ndarray:
    data = scale_iris_dataset(data)
    return data


@task
def train_test_split_iris_data(
    data: ndarray, target: ndarray
) -> (ndarray, ndarray, ndarray, ndarray):
    train_data, val_data, train_target, val_target = train_test_data_split(
        data=data, target=target
    )
    return train_data, val_data, train_target, val_target


@task(cache=True, cache_version="1.0")
def train_iris_data(
    train_data: ndarray, train_target: ndarray
) -> IrisClassificationModel:
    model = train_iris_dataset(train_data=train_data, train_target=train_target)
    return model


@task
def test_iris_model(
    model: IrisClassificationModel, data: ndarray, target: ndarray, accuracy_type: str
) -> pd.DataFrame:
    accuracy = calculate_accuracy(
        model=model, data=data, target=target, accuracy_type=accuracy_type
    )
    return accuracy


@task
def save_model(model: IrisClassificationModel, artifact_name: str) -> None:
    bentoml.pytorch.save(name=artifact_name, model=model)


@task
def test_model_deployment(artifact_name: str) -> None:
    test_runner = bentoml.pytorch.load_runner(tag=artifact_name)
    x = Variable(torch.FloatTensor([5.9, 3.0, 5.1, 1.8]))
    print(test_runner.run(x))


@workflow
def my_wf() -> (pd.DataFrame, pd.DataFrame):

    # take care of keywords
    # flyte takes that seriously

    artifact_name = "iris_classifier"

    data, target, target_names, feature_names = get_iris_data()
    scaled_data = scale_iris_data(data=data)
    train_data, val_data, train_target, val_target = train_test_split_iris_data(
        data=scaled_data, target=target
    )
    model = train_iris_data(train_data=train_data, train_target=train_target)
    _train_accuracy = test_iris_model(
        model=model, data=train_data, target=train_target, accuracy_type="train"
    )
    _test_accuracy = test_iris_model(
        model=model, data=val_data, target=val_target, accuracy_type="test"
    )

    save_model(model=model, artifact_name=artifact_name)
    test_model_deployment(artifact_name=artifact_name)

    return _train_accuracy, _test_accuracy


if __name__ == "__main__":
    train_accuracy, test_accuracy = my_wf()

    wandb.log({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy})
