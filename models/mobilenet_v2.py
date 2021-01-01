import torch.nn as nn
import torchvision.models as models


def Net(num_classes: int) -> nn.Module:
    net = models.mobilenet_v2(pretrained=True)
    net.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(net.last_channel, num_classes),
    )
    return net
