import torch
from model import IrisClassificationModel
import numpy as np
from torch import nn


def train_dataset(train_data, train_target):

    #Define training hyperprameters.
    batch_size = 60
    num_epochs = 500
    learning_rate = 0.01
    size_hidden = 100

    #Calculate some other hyperparameters based on data.
    batch_no = len(train_data) // batch_size  #batches
    cols = train_data.shape[1] #Number of columns in input matrix
    classes = len(np.unique(train_target))

    model = IrisClassificationModel(input_dim=cols)


    #Adam is a specific flavor of gradient decent which is typically better
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

        print('Epoch {}'.format(epoch + 1), "loss: ", running_loss)
        running_loss: float = 0.0
