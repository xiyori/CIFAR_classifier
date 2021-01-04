import torch
import models.model_conv_best as model
import log_utils.log_tensorboard as log
# import training.scheduler as scheduler
from training.train import train
import training.dataset as ds
from training.validation import test


if __name__ == "__main__":
    log.init("lel_kek")

    # Try to use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create an instance of the model
    net = model.Net()
    net.to(device)
    # PATH = 'model_instances/cifar_net_79%_best.pth'
    # net.load_state_dict(torch.load(PATH))

    train(net, epoch_count=20, start_epoch=10, use_scheduler=False)
    # test(net)

    # Save our beautiful model for future generations
    # PATH = 'model_instances/cifar_net_tmp.pth'
    # torch.save(net.state_dict(), PATH)
