import torch.nn as nn
import torch.nn.functional as F


class IrisClassificationModel(nn.Module):
    def __init__(self, input_dim: int, hidden_features: int = 50):
        super(IrisClassificationModel, self).__init__()
        self.layer1 = nn.Linear(in_features=input_dim, out_features=hidden_features)
        self.layer2 = nn.Linear(
            in_features=hidden_features, out_features=hidden_features
        )
        self.layer3 = nn.Linear(in_features=hidden_features, out_features=3)

    def forward(self, x):
        x = F.relu(input=self.layer1(x))
        x = F.dropout(input=x, p=0.1)
        x = F.relu(input=self.layer2(x))
        x = F.softmax(input=self.layer3(x), dim=1)
        return x
