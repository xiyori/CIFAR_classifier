import torch
import torch.nn as nn
import dataset as ds
import algorithm
import scheduler
import log
import validation


# Train model for 'epoch_count' epochs
def train(net: nn.Module, epoch_count: int) -> None:
    criterion = algorithm.get_loss()
    for data_iter in range(epoch_count):
        optimizer = algorithm.get_optimizer(net,
                                            scheduler.params_list[data_iter])

        average_loss = 0.0
        correct = 0
        total = 0
        for _, data in enumerate(ds.trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)

            # stats
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            average_loss += loss.item()

        average_loss /= 50000
        train_accuracy = 100 * correct // total
        test_accuracy = validation.test(net)
        log.add((train_accuracy, test_accuracy, average_loss))
        print('[%d, %5d] average loss: %.3f' %
              (data_iter + 1, total, average_loss))
        print('Train accuracy: %d %%' % train_accuracy)
        print('Test accuracy: %d %%' % test_accuracy)

        PATH = 'model/net_tmp_epoch_%d_acc_%d.pth' % (data_iter + 1, test_accuracy)
        torch.save(net.state_dict(), PATH)
        log.save()
    print('Complete')
