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

loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr = 0.001)
