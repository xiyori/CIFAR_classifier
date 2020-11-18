import torch
import torchvision
import model_conv_overfit as model
import torch.optim as optim
import torch.nn as nn
import dataset as ds

if __name__ == "__main__":
    # Shows some images in the dataset
    # dataiter = iter(ds.trainloader)
    # images, labels = dataiter.next()
    #
    # ds.imshow(torchvision.utils.make_grid(images))
    # print(' '.join((ds.classes[labels[j]] for j in range(ds.batch_count))))

    # Try to use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create an instance of the model
    net = model.Net()
    net.cuda()
    # PATH = './cifar_net_tmp.pth'
    # net.load_state_dict(torch.load(PATH))

    # Use gradient descent for learning
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=0.9)

    # Train model for 'epoch_count' epochs
    epoch_count = 1
    for data_iter in range(epoch_count):
        average_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(ds.trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            average_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] average loss: %.3f' %
                      (data_iter + 1, i + 1, average_loss / 2000))
                average_loss = 0.0
        print('Train accuracy: %d %%' % (
                100 * correct / total))
        # if (data_iter + 1) % 5 == 0:
        #     PATH = './cifar_net_tmp_%d.pth' % (data_iter + 1)
        #     torch.save(net.state_dict(), PATH)
    print('Complete')

    # Save our beautiful model for future generations
    PATH = './cifar_net_tmp.pth'
    torch.save(net.state_dict(), PATH)
