'''Train CIFAR10 with PyTorch.'''
# Code Attribution: https://github.com/kuangliu/pytorch-cifar
import torch
import torch.nn as nn
import torch.optim as optim
from optim import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from prettytable import PrettyTable
from utils import progress_bar

import time
import os
import argparse

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('optimizer', default='SGD', type=str, action='store_const', 
                    help='optimizer')
parser.add_argument('dataloader_workers', default=4, type=int, action='store_const',
                    help='dataloader workers')
parser.add_argument('epochs', default=5, type=int, action='store_const',
                    help='epochs')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate', dest=lr)
parser.add_argument('--data_path', default='./data', type=str, action='store_const',
                    help='data path')
parser.add_argument('--cuda', default=False, type=str, action='store_true',
                    help='cuda usage')

parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
ops = ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
assert(args.optimizer in ops),f'Invalid optimizer...\n Must be in {ops}'
device = 'cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=args.dataloader_workers)

testset = torchvision.datasets.CIFAR10(
    root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=args.dataloader_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = args.optimizer(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


###C1###

outfile = open('C1.txt', 'w')
train_times = []
test_times = []
total_times = []

for epoch in range(args.epochs):
    start = time.perf_counter()
    train(epoch)
    train_time = time.perf_counter()
    test(epoch)
    test_time = time.perf_counter()
    
    train_times.append(train_time - start)
    test_times.append(test_time - train_time)
    total_times.append(test_time - start)
    scheduler.step()

table = PrettyTable([])
table.add_column('epoch', range(args.epochs))
table.add_column('train_time', train_times)
table.add_column('test_time', test_times)

print(table, file=outfile)
print("Average train time: ", sum(train_times) / len(train_times), file=outfile)
print("Average test time: ", sum(test_times) / len(test_times), file=outfile)
print("Average total time: ", sum(total_times) / len(total_times), file=outfile)
outfile.close()
