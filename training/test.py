import torch
import torch.nn as nn
import training.dataset as ds


def test(net: nn.Module) -> int:
    correct = 0
    total = 0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    with torch.no_grad():
        for data in ds.testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(ds.batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = 100 * correct // total
    print('Accuracy on 10000 test images: %d %%' % accuracy)

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            ds.classes[i], 100 * class_correct[i] / class_total[i]))
    return accuracy
