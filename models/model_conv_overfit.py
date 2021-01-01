import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as Tensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 56, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.drop = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(56, 112, 5)
        # self.norm = nn.BatchNorm1d(112 * 5 * 5)
        self.fc1 = nn.Linear(112 * 5 * 5, 894)
        self.fc2 = nn.Linear(894, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 112 * 5 * 5)
        # x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
