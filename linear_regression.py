import torch
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

def adjust_learning_rate(optimizer, epoch, init_lr):
    lr = init_lr * (0.1 ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

num_inputs = 2
num_examples = 1000

true_w = [1, 2]
true_b = 3

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs) ), dtype = torch.float )
labels = features[:,0] * true_w[0] + features[:,1] * true_w[1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size = labels.size()), dtype = torch.float )

batch_size = 10

dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size, shuffle = True)

net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
print(net)

print('*'*20)
print('parameters before initialization')
print('*'*20)
for param in net.parameters():
    print(param)
print('*'*20)

init.normal_(net[0].weight, mean = 0, std = 0.1)
init.constant_(net[0].bias, val = 0.5)

print('parameters after initialization')
print('*'*20)
for param in net.parameters():
    print(param)

loss = nn.MSELoss()

init_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr = init_lr)
print(optimizer)

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    # adjust_learning_rate(optimizer, epoch, init_lr)
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch: ', epoch, ', loss: ', l.item())

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
