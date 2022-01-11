import numpy as np
from mil.data.datasets import mnist_bags
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as Data
from torch.utils.data import  TensorDataset


# normalize type
torch.set_default_tensor_type(torch.DoubleTensor)


# hyper parameters
EPOCH = 1  # to save time, only run 1 time
BATCH_SIZE = 1
LR = 0.0005


# pre-processing
# input data
""" bags_train: matrix of bag put in a list
    y_train: label of each instance put in a list 
    train_ins: matrix of each instance put in a list  """
(bags_train, y_train, train_ins), (bags_test, y_test, test_ins) = mnist_bags.load()
bags_train_1D =[np.array(bag).reshape(-1, 28*28) for bag in bags_train]
bags_test_1D =[np.array(bag).reshape(-1, 28*28) for bag in bags_test]
# there are different quantities of patches in different bag
pad_train = pad_sequence([torch.tensor(dim) for dim in bags_train_1D], batch_first=True)
pad_test = pad_sequence([torch.tensor(dim) for dim in bags_test_1D], batch_first=True)
# transform to tensor
xtrain1 = torch.tensor(pad_train)
y_train = torch.tensor(y_train)
xtest1 = torch.tensor(pad_test)
y_test = torch.tensor(y_test)
# get right shape
x_train = xtrain1.unsqueeze(1)
x_test = xtest1.unsqueeze(1)
# create dataset
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)
# load data
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
test_loader = Data.DataLoader(dataset = test_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)


# building net
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                # bag size 1*200*12544
                in_channels = 1,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                padding = 2  # pad=(kernel_size-1)/2
            ),  # bag size 16*200*12544
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*4*196, 2)  # I used x.shape() to know current shape, then deleted it.

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32*4*196)
        output = self.out(x)
        return output


# train
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func = torch.nn.CrossEntropyLoss()  # classification
for epoch in range (EPOCH):
    for i, data in enumerate(train_loader):
        output = cnn(x_train)
        loss = loss_func(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # test and output the result every 50 step
        if i % 50 == 0:
            test_output = cnn(x_test)
            y_pred = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(y_pred == y_test) / y_test.size(0)
            print('Epoch: ', epoch)
            print('train loss: %.4f'%loss.item())
            print('test accuracy: %.4f'%accuracy)





