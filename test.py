import torch
import torchvision
import model_conv_best as model
import dataset as ds

if __name__ == "__main__":
    # Use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Let's test our NN
    # Load trained model
    net = model.Net()
    net.cuda()
    PATH = './cifar_net_72%.pth'
    net.load_state_dict(torch.load(PATH))

    # Show test batch
    # dataiter = iter(ds.testloader)
    # images, labels = dataiter.next()
    # images = images.cuda()
    # labels = labels.cuda()
    #
    # ds.imshow(torchvision.utils.make_grid(images))
    # print("Actual labels:", ' '.join((ds.classes[labels[j]]
    #                                   for j in range(ds.batch_count))))
    #
    # # Show NN output
    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)
    #
    # print('Predicted:', ' '.join('%5s' % ds.classes[predicted[j]]
    #                              for j in range(ds.batch_count)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in ds.testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on 10000 test images: %d %%' % (
            100 * correct / total))
