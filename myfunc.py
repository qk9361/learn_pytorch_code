import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import torch.utils.data as Data
from collections import OrderedDict
# from IPython import display

# def set_figsize(figsize=(3.5, 2.5)):
#     use_svg_display()
#     # 设置图的尺寸
#     plt.rcParams['figure.figsize'] = figsize
#
# def use_svg_display():
#     """Use svg format to display plot in jupyter"""
#     display.set_matplotlib_formats('svg')

# This function is used to download data by using torchvision. If data exists, this function
# will direct load it. After loading data successfully, small batches will be generated.
def load_data_fashion_mnist(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root = 'Datasets/FashionMNIST', train = True, download =
    True, transform = transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root = 'Datasets/FashionMNIST', train = False, download = True,
    transform = transforms.ToTensor())
    print(mnist_train)
    print('number of training samples: ', len(mnist_train), '\n',
    'number of testing samples: ', len(mnist_test))

    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4

    train_iter = Data.DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    test_iter = Data.DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = num_workers)

    return train_iter, test_iter

# This function helps to transform num_label to real labels.
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize = (12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28,28)).numpy() )
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def get_num_inputs(train_iter):
    X, y = iter(train_iter).next()
    num_inputs = len(X[0].view(-1, 1))
    return num_inputs

# For the reshape of the training data
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

# For evalution of the accuracy in test data
def evaluate_accuracy(test_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in test_iter:
        acc_sum += (net(X).argmax(dim = 1) == y).sum().item()
        n += y.shape[0]
    return acc_sum / n

# Define how to train a net
def train_net(net, train_iter, test_iter, loss, num_epochs, batch_size,
params = None, lr = None, optimizer = None):
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        net.train()
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                param.grad.data.zero_()

            l.backward()
            optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]
        net.eval()
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f' %(epoch,
        train_l_sum / n, train_acc_sum / n, test_acc, time.time() - start))

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
                                nn.Conv2d(1, 6, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2,2),
                                nn.Conv2d(6, 16, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2,2),
                                )
        self.fc = nn.Sequential(
                                nn.Linear(16*5*5, 120),
                                nn.ReLU(),
                                nn.Linear(120, 84),
                                nn.ReLU(),
                                nn.Linear(84, 10)
                                )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output 
