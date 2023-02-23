"""Train CIFAR10 with PyTorch."""
# Code Attribution: https://github.com/kuangliu/pytorch-cifar
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from prettytable import PrettyTable
from utils import print_config, set_optimizer

import time
import os
import argparse

from models import *

EXERCISES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "Q3"]

def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument("exercise", default="C1", help="problem # on HW")
    parser.add_argument("--epochs", default=5, type=int, help="num epochs; default 5")
    parser.add_argument("--optimizer", default="SGD", help="optimizer, default SGD")
    parser.add_argument(
        "--dataloader_workers", default=4, help="dataloader workers; default 4"
    )
    parser.add_argument(
        "--data_path", default="./data", help="data dirpath; default ./data"
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate, default 0.1")
    parser.add_argument("--cuda", default=False, help="cuda usage; default False")
    parser.add_argument(
        "--resume", "-r", action="store_true", default=False,
        help="resume from checkpoint; default False",
    )

    args = parser.parse_args()

    if args.exercise not in EXERCISES:
        raise ValueError(f"Invalid exercise \n Must be in {EXERCISES}")
    device = "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    args.__setattr__("device", device)
    args = set_optimizer(args)
    print("==> Setting configs..")

    print_config(args, device)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print("==> Preparing data..")
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

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # Model
    print("==> Building model..")
    net = ResNet18()
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load("./checkpoint/ckpt.pth")
        net.load_state_dict(checkpoint["net"])
        best_acc = checkpoint["acc"]
        start_epoch = checkpoint["epoch"]

    criterion = nn.CrossEntropyLoss()
    optimizer = args.optimizer_function(
        net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print("\nEpoch: %d" % epoch)
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

            # progress_bar(
            #     batch_idx,
            #     len(trainloader),
            #     "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
            #     % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            # )


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

                # progress_bar(
                #     batch_idx,
                #     len(testloader),
                #     "Test Loss: %.3f | Acc: %.3f%% (%d/%d)"
                #     % (
                #         test_loss / (batch_idx + 1),
                #         100.0 * correct / total,
                #         correct,
                #         total,
                #     ),
                # )

        # Save checkpoint.
        acc = 100.0 * correct / total
        if acc > best_acc:
            print("Saving..")
            state = {
                "net": net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/ckpt.pth")
            best_acc = acc


    ###C2###

    if args.exercise == "C2":
        outfile = open("C2.txt", "w")
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
        table.add_column("epoch", range(args.epochs))
        table.add_column("train_time", train_times)
        table.add_column("test_time", test_times)

        print(table, file=outfile)
        print(
            "Average train time per epoch: ",
            sum(train_times) / len(train_times),
            file=outfile,
        )
        print(
            "Average test time per epoch: ", sum(test_times) / len(test_times), file=outfile
        )
        print(
            "Average total time epoch: ", sum(total_times) / len(total_times), file=outfile
        )
        print(f"Total time for {args.epochs} epochs: ", sum(total_times), file=outfile)
        outfile.close()


if __name__ == '__main__':
    main()