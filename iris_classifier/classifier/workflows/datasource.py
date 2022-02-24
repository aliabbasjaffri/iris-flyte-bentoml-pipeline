import numpy as np
from numpy import ndarray
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_iris_dataset() -> (ndarray, ndarray, ndarray, ndarray):
    """

    :return:

    """
    iris = load_iris()
    return iris.data, iris.target, iris.target_names, iris.feature_names


def scale_iris_dataset(data: ndarray) -> ndarray:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def train_test_data_split(
    data: ndarray, target: ndarray
) -> (ndarray, ndarray, ndarray, ndarray):
    train_data, val_data, train_target, val_target = train_test_split(
        data, target, train_size=0.8, test_size=0.2, random_state=123, stratify=target
    )

    print("All:", np.bincount(target) / float(len(target)) * 100.0)
    print("Training:", np.bincount(train_target) / float(len(train_target)) * 100.0)
    print("Validation:", np.bincount(val_target) / float(len(val_target)) * 100.0)

    return train_data, val_data, train_target, val_target
