import torch
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray
from torch.autograd import Variable

artifact_name = "iris_classifier"

iris_classifier_runner = bentoml.pytorch.load_runner(tag=artifact_name)
svc = bentoml.Service(name=artifact_name, runners=[iris_classifier_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    x = Variable(torch.FloatTensor(input_series))
    return iris_classifier_runner.run(x)
