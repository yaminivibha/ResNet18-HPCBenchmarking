"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
# Code Attribution: https://github.com/kuangliu/pytorch-cifar
"""
import os
import sys
import time
import math
import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
from torch.optim import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax

OPTIMIZERS = ["SGD", "Adam", "RMSprop", "Adagrad", "Adadelta", "Adamax"]

def print_config(args, device):
    """Print config."""
    print(f"Exercise:               {args.exercise}")
    print(f"Epochs:                 {args.epochs}")
    print(f"Optimizer:              {args.optimizer}")
    print(f"Num Dataloader Workers: {args.dataloader_workers}")
    print(f"Data Path:              {args.data_path}")
    print(f"Learning Rate:          {args.lr}")
    print(f"Device:                 {device}")
    # print(f"Resume:                 {args.resume}")


def set_optimizer(args):
    """Set optimizer."""
    if args.optimizer == "SGD":
        optimizer = SGD
    elif args.optimizer == "Adam":
        optimizer = Adam
    elif args.optimizer == "RMSprop":
        optimizer = RMSprop
    elif args.optimizer == "Adagrad":
        optimizer = Adagrad
    elif args.optimizer == "Adadelta":
        optimizer = Adadelta
    elif args.optimizer == "Adamax":
        optimizer = Adamax
    else:
        raise ValueError(f"Invalid optimizer \n Must be in {OPTIMIZERS}")
    args.__setattr__("optimizer_function", optimizer)
    return args

def load_data(args):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args.dataloader_workers
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.dataloader_workers
    )
    return trainloader, trainset, testloader, testset


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


term_width = 80

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f
