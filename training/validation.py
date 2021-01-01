import torch
import models.model_conv_best as model
import training.dataset as ds
import training.algorithm as algorithm
import numpy as np


def test(net: model.Net) -> (int, float, np.array):
    criterion = algorithm.get_loss()
    average_loss = 0.0
    correct = 0
    total = len(ds.testloader.dataset)
    conf_matrix = np.zeros((len(ds.classes), len(ds.classes)))
    with torch.no_grad():
        for data in ds.testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            average_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                conf_matrix[labels[i], predicted[i]] += 1.0
    return 100 * correct // total, average_loss / total, conf_matrix
