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

batch_size = 256
train_iter, test_iter = mf.load_data_fashion_mnist(batch_size)

num_inputs = mf.get_num_inputs(train_iter)
num_outputs = 10

from collections import OrderedDict
net = nn.Sequential(
                    OrderedDict([
                    ('flatten', mf.FlattenLayer()),
                    ('linear', nn.Linear(num_inputs, num_outputs))
                                ])
                    )

init.normal_(net.linear.weight, mean = 0, std = 0.1)
init.constant_(net.linear.bias, val = 0.5)

loss = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr = 0.1)
num_epochs = 10

mf.train_net(net, train_iter, test_iter, loss, num_epochs, batch_size,
None, None, optimizer)

X, y = iter(test_iter).next()

true_labels = mf.get_fashion_mnist_labels(y.numpy())
pred_labels = mf.get_fashion_mnist_labels(net(X).argmax(dim = 1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
mf.show_fashion_mnist(X[0:9], titles[0:9])
