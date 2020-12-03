import torch
import torch.nn.functional as F
import numpy as np
import dataset as ds
import model_conv_overfit
import model_conv_fair
import model_conv_best


if __name__ == "__main__":
    # Try to use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    nets = []
    weights = np.array([72, 77, 79], dtype=float)

    net = model_conv_fair.Net()
    net.to(device)
    PATH = 'model/cifar_net_72%_fair.pth'
    net.load_state_dict(torch.load(PATH))
    nets.append(net)

    net = model_conv_overfit.Net()
    net.to(device)
    PATH = 'model/cifar_net_77%_over.pth'
    net.load_state_dict(torch.load(PATH))
    nets.append(net)

    net = model_conv_best.Net()
    net.to(device)
    PATH = 'model/cifar_net_79%_best.pth'
    net.load_state_dict(torch.load(PATH))
    nets.append(net)

    weights /= weights.sum()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in ds.testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = torch.tensor([[0 for _ in range(10)] for __ in range(4)],
                                   dtype=float).cuda()
            for i in range(3):
                outputs += F.softmax(net(images)) * weights[i]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(100 * correct // total)
