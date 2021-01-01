import torch.optim as optim
import torch.nn as nn


def get_loss() -> nn.Module:
    return nn.CrossEntropyLoss()


def get_optimizer(net: nn.Module, params: list=tuple()) -> optim.Optimizer:
    # return optim.SGD(net.parameters(), lr=params[0], momentum=params[1])
    return optim.Adam(net.parameters())


def update_optimizer(optimizer: optim.Optimizer, params: list) -> None:
    # for g in optimizer.param_groups:
    #     g['lr'] = params[0]
    #     g['momentum'] = params[1]
    pass
