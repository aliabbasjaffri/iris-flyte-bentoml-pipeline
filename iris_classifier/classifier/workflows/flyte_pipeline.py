"""
Flyte Pipeline for Iris Dataset classification
------------
"""
from numpy import ndarray
from train import train_iris_dataset
from flytekit import task, workflow
from model import IrisClassificationModel
from datasource import load_iris_dataset, scale_iris_dataset, train_test_data_split


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
def train_iris_data(train_data: ndarray, train_target: ndarray) -> IrisClassificationModel:
    model = train_iris_dataset(train_data=train_data, train_target=train_target)
    return model


@workflow
def my_wf() -> IrisClassificationModel:

    # take care of keywords
    # flyte takes that seriously

    data, target, target_names, feature_names = get_iris_data()
    scaled_data = scale_iris_data(data=data)
    train_data, val_data, train_target, val_target = train_test_split_iris_data(
        data=scaled_data, target=target
    )
    model = train_iris_data(train_data=train_data, train_target=train_target)
    return model


if __name__ == "__main__":
    my_wf()
