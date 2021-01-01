import torch
import torch.nn as nn
import training.dataset as ds
import training.algorithm as algorithm
import training.scheduler as scheduler
import training.validation as validation
import log_utils.log_tensorboard as log
import log_utils.plot_utils as plot_utils
from progress.bar import IncrementalBar


# Train model for 'epoch_count' epochs
def train(net: nn.Module, epoch_count: int, start_epoch: int=0,
          use_scheduler: bool=False) -> None:
    criterion = algorithm.get_loss()
    if use_scheduler:
        optimizer = algorithm.get_optimizer(net,
                                            scheduler.params_list[start_epoch])
    else:
        optimizer = algorithm.get_optimizer(net)
    bar_step = 1000

    for epoch_idx in range(start_epoch, epoch_count):
        net.train()

        if use_scheduler:
            algorithm.update_optimizer(optimizer,
                                       scheduler.params_list[epoch_idx])

        average_loss = 0.0
        correct = 0
        total = len(ds.trainloader.dataset)
        curr_iter = 0
        iter_bar = IncrementalBar("Current progress", max=total,
                                  suffix='%(percent)d%%')
        
        for _, data in enumerate(ds.trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)

            # stats
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            average_loss += loss.item()

            if curr_iter >= bar_step:
                iter_bar.next(bar_step)
                curr_iter -= bar_step
            curr_iter += ds.batch_size

        iter_bar.goto(total)
        iter_bar.finish()

        average_loss /= total
        train_accuracy = 100 * correct // total

        net.eval()
        test_accuracy, test_loss, conf_matrix = validation.test(net)

        log.add(epoch_idx, (train_accuracy, test_accuracy,
                            average_loss, test_loss),
                (plot_utils.mat_to_img(conf_matrix), ))
        print('[%d, %5d] average loss: %.3f, test loss: %.3f' %
              (epoch_idx, total, average_loss, test_loss))
        print('Train accuracy: %d %%' % train_accuracy)
        print('Test accuracy: %d %%' % test_accuracy)

        PATH = 'model_instances/net_tmp_epoch_%d_acc_%d.pth' % (epoch_idx, test_accuracy)
        torch.save(net.state_dict(), PATH)
        log.save()
    print('Complete')
