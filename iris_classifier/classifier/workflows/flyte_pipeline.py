"""
Flyte Pipeline for Iris Dataset classification
------------
"""
from numpy import ndarray
from torch import nn
from train import train_iris_dataset
from flytekit import task, workflow
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
def train_iris_data(
    train_data: ndarray, train_target: ndarray
) -> nn.Module:
    model = train_iris_dataset(train_data=train_data, train_target=train_target)
    return model


# %%
# Here we declare a workflow called ``my_wf``. Note the @workflow decorator,
# Flyte finds all workflows that you have declared by finding this decorator.
# A @workflow function, looks like a regular python function, except for some
# important differences, it is never executed by classifier-engine. It is like
# psuedo code, that is analyzed by flytekit to convert to Flyte's native
# Workflow representation. Thus the variables like return values from `tasks`
# are not real values, and trying to interact with them like regular variables
# will result in an error. For example, if a task returns a boolean, and if you
# try to test the truth value for this boolean, an error will be raised. The
# reason, is the tasks are not really executed by the function, but run remote
# and the return variables are supplied to subsequent tasks.
#
# You can treat the outputs of a task as you normally would a Python function. Assign the output to two variables
# and use them in subsequent tasks as normal. See :py:func:`flytekit.workflow`
# You can change the signature of the workflow to take in an argument like this:
@workflow
def my_wf() -> nn.Module:

    data, target, target_names, feature_names = get_iris_data()
    data = scale_iris_data(data)
    train_data, val_data, train_target, val_target = train_test_split_iris_data(
        data, target
    )
    model = train_iris_data(train_data, train_target)
    return model


if __name__ == "__main__":
    my_wf()
