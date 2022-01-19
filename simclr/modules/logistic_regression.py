import torch
import torch.nn as nn
import torch.nn.functional as F


#class LogisticRegression(nn.Module):
#    def __init__(self, n_features, n_classes):
#        super(LogisticRegression, self).__init__()
#
#        self.fc1 = nn.Linear(n_features, n_features)
#        self.fc2 = nn.Linear(n_features, n_classes)
#
#    def forward(self, x):
#        return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))

# https://github.com/martinmamql/fair-mixup/blob/master/celeba/model.py
class LinearModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_features, n_features)
        self.fc2 = nn.Linear(n_features, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        outputs = self.fc2(x)
        return torch.sigmoid(outputs)

