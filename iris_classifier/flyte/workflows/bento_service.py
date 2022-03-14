try:
    from datasource import load_iris_dataset
except ImportError:
    from .datasource import load_iris_dataset
import torch
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray
from torch.autograd import Variable

artifact_name = "iris_classifier"
_, _, target_names, _ = load_iris_dataset()
iris_classifier_runner = bentoml.pytorch.load_runner(tag=artifact_name)
svc = bentoml.Service(name=artifact_name, runners=[iris_classifier_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> str:
    x = Variable(torch.FloatTensor(input_series))
    prediction = iris_classifier_runner.run(x)
    value = target_names[np.where(prediction == 1.0)[0]]
    return value
