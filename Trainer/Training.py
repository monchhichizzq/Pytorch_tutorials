# -*- coding: utf-8 -*-
# @Time    : 2020/9/5 22:39
# @Author  : Zeqi@@
# @FileName: Trainer.py
# @Software: PyCharm

import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.ResNet_cifar import *

from data_process.data_loader import data_generator, unpickle_cifar
from Trainer.utils import acc_monitoring

import numpy as np
from PIL import Image


def preparing_data():
    print('==> Preparing data..')
    cifar_train, cifar_test = unpickle_cifar()
    x_train, x_test, y_train, y_test, classnames = data_generator(cifar_train, cifar_test, is_normalized=False)

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToPILImage()
        # ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.ToPILImage()
    ])

    x_train = [transform_train(x).numpy() for x in x_train]
    x_test = [transform_test(x).numpy() for x in x_test]
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)

    trainset = torch.utils.data.TensorDataset(x_train,  torch.from_numpy(y_train))
    testset = torch.utils.data.TensorDataset(x_test, torch.from_numpy(y_test))

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False)
    return trainloader, testloader, classnames

def load_model():
    # Model
    print('==> Building model..ResNet 18')
    net = ResNet18()
    net = net.to(device)
    summary(net, input_size=(3, 32, 32))
    return net

def train(epoch, model, loader):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_loss_ = train_loss/(batch_idx+1)
        train_acc_ = 100.*correct/total

    print(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return train_loss_, train_acc_, model

def test(model, loader):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # when in test stage, no grad
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            test_loss_ = test_loss / (batch_idx + 1)
            test_acc_ = 100. * correct / total

    print(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return test_loss_, test_acc_





if __name__ == '__main__':
    lr = 1e-2
    start_epoch = 0
    total_epoch = 500

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    trainloader, testloader, classnames = preparing_data()
    net =load_model()

    #
    #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)

    epochs, train_losses, train_accs, test_losses, test_accs = [], [], [], [], []
    for epoch in range(start_epoch, start_epoch + total_epoch):
        net = trained_net if epoch>1 else net
        train_loss, train_acc, trained_net = train(epoch, net, trainloader)
        test_loss, test_acc = test(trained_net, testloader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        epochs.append(epoch+1)

        acc_monitoring(epochs,
                       train_losses,
                       test_losses,
                       train_accs,
                       test_accs,
                       save_path = 'board.png')
    #
