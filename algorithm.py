import torch.optim as optim
import torch.nn as nn


def get_loss() -> nn.Module:
    return nn.CrossEntropyLoss()


def get_optimizer(net: nn.Module, params: list) -> optim.Optimizer:
    return optim.SGD(net.parameters(), lr=params[0], momentum=params[1])
