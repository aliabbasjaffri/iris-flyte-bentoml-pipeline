try:
    from model import IrisClassificationModel
    from datasource import load_iris_dataset, scale_iris_dataset, train_test_data_split
except ImportError:
    from .model import IrisClassificationModel
    from .datasource import load_iris_dataset, scale_iris_dataset, train_test_data_split
import torch
import numpy as np
from torch import nn
from numpy import ndarray


def train_iris_dataset(
    train_data: ndarray, train_target: ndarray
) -> IrisClassificationModel:

    # Define training hyperprameters.
    batch_size = 60
    num_epochs = 500
    learning_rate = 0.01
    size_hidden = 100

    # Calculate some other hyperparameters based on data.
    batch_no = len(train_data) // batch_size  # batches
    cols = train_data.shape[1]  # Number of columns in input matrix
    classes = len(np.unique(train_target))

    model = IrisClassificationModel(input_dim=cols)

    # Adam is a specific flavor of gradient decent which is typically better
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    from sklearn.utils import shuffle
    from torch.autograd import Variable

    running_loss = 0.0
    for epoch in range(num_epochs):
        # Shuffle just mixes up the dataset between epocs
        train_data, train_target = shuffle(train_data, train_target)

        # Mini batch learning
        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size

            inputs = Variable(torch.FloatTensor(train_data[start:end]))
            labels = Variable(torch.LongTensor(train_target[start:end]))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print("Epoch {}".format(epoch + 1), "loss: ", running_loss)
        running_loss: float = 0.0

    return model


if __name__ == "__main__":
    # testing train_iris_dataset

    data, target, _, _ = load_iris_dataset()
    data = scale_iris_dataset(data)
    train_data, val_data, train_target, val_target = train_test_data_split(data=data, target=target)
    train_iris_dataset(train_data=train_data, train_target=train_target)
