import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys
import torch.utils.data as Data
import myfunc as mf
from collections import OrderedDict

num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
                OrderedDict([
                ('flatten', mf.FlattenLayer()),
                ('linear1', nn.Linear(num_inputs, num_hiddens)),
                ('avtivation', nn.ReLU()),
                ('linear2', nn.Linear(num_hiddens, num_outputs))
                ])
                )

for param in net.parameters():
    print(param.shape)
    init.normal_(param, mean = 0, std = 0.1)

batch_size = 256
train_iter, test_iter = mf.load_data_fashion_mnist(batch_size)

loss = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr = 0.5)

num_epochs = 10
mf.train_net(net, train_iter, test_iter, loss, num_epochs, batch_size,
None, None, optimizer)

X, y = iter(test_iter).next()

true_labels = mf.get_fashion_mnist_labels(y.numpy())
pred_labels = mf.get_fashion_mnist_labels(net(X).argmax(dim = 1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
mf.show_fashion_mnist(X[0:9], titles[0:9])
