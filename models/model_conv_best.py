import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as Tensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 112, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(112, 56, 5, padding=2)
        self.conv3 = nn.Conv2d(56, 164, 5, padding=3)
        self.conv4 = nn.Conv2d(164, 64, 3, padding=2)
        self.fc1 = nn.Linear(64 * 3 * 3, 192)
        self.fc2 = nn.Linear(192, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
