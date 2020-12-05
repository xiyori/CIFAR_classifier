import torch
import model_conv_best as model
import dataset as ds
import algorithm


def test(net: model.Net) -> (int, float):
    criterion = algorithm.get_loss()
    average_loss = 0.0
    correct = 0
    total = len(ds.testloader.dataset)
    with torch.no_grad():
        for data in ds.testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            average_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    return 100 * correct // total, average_loss / total
