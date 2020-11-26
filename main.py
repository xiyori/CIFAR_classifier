import torch
import model_600 as model
import log
import scheduler
from train import train
from test import test


if __name__ == "__main__":
    # Try to use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create an instance of the model
    net = model.Net()
    net.to(device)
    PATH = 'model/cifar_net_56%_600.pth'
    net.load_state_dict(torch.load(PATH))

    # train(net, epoch_count=scheduler.count_epoch())
    test(net)

    # Save our beautiful model for future generations
    # PATH = 'model/cifar_net_tmp.pth'
    # torch.save(net.state_dict(), PATH)

    # log.plot()


