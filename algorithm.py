import torch.optim as optim
import torch.nn as nn
import loss


def get_loss() -> nn.Module:
    return loss.BestCrossEntropyLossEver()


def get_optimizer(net: nn.Module, params: list) -> optim.Optimizer:
    return optim.SGD(net.parameters(), lr=params[0], momentum=params[1])
