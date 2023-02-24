"""Train CIFAR10 with PyTorch."""
# Code Attribution: https://github.com/kuangliu/pytorch-cifar
import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from prettytable import PrettyTable
from torch.optim import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop
from torch.utils.benchmark import Timer

from models import *
from utils import load_data, print_config, progress_bar, set_optimizer

EXERCISES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "Q3"]
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
    parser.add_argument(
        "--lr", default=0.1, type=float, help="learning rate, default 0.1"
    )
    parser.add_argument(
        "--outfilename", default="output.txt", help="filename for writing results"
    )
    parser.add_argument("--cuda", default=True, help="cuda usage; default False")
    # parser.add_argument(
    #     "--resume",
    #     "-r",
    #     action="store_true",
    #     default=False,
    #     help="resume from checkpoint; default False",
    # )

    args = parser.parse_args()

    print("==> Setting configs..")
    if args.exercise not in EXERCISES:
        raise ValueError(f"Invalid exercise \n Must be in {EXERCISES}")
    args.device = "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    args.optimizer = set_optimizer(args)
    print_config(args)

    # best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print("==> Preparing data..")
    trainloader, trainset, testloader, testset = load_data(args)

    # Model
    print("==> Building model..")
    net = ResNet18()
    net = net.to(args.device)
    if args.device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # if args.resume:
    #     # Load checkpoint.
    #     print("==> Resuming from checkpoint..")
    #     assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    #     checkpoint = torch.load("./checkpoint/ckpt.pth")
    #     net.load_state_dict(checkpoint["net"])
    #     best_acc = checkpoint["acc"]
    #     start_epoch = checkpoint["epoch"]

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

        c2_load_time = 0
        c2_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            c2_load_time += time.time() - c2_start
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(trainloader),
                "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
            c2_start = time.time()
        return c2_load_time

    def test(epoch):
        # global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            c2_load_time = 0
            c2_start = time.time()
            for batch_idx, (inputs, targets) in enumerate(testloader):
                c2_load_time += time.time() - c2_start
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(
                    batch_idx,
                    len(testloader),
                    "Test Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )
                c2_start = time.time()
        # Save checkpoint.
        # acc = 100.0 * correct / total
        # if acc > best_acc:
        #     print("Saving..")
        #     state = {
        #         "net": net.state_dict(),
        #         "acc": acc,
        #         "epoch": epoch,
        #     }
        #     if not os.path.isdir("checkpoint"):
        #         os.mkdir("checkpoint")
        #     torch.save(state, "./checkpoint/ckpt.pth")
        #     best_acc = acc
        return c2_load_time

    ###C2: Time Measurement###

    if args.exercise == "C2":
        outfile = open("C2.txt", "w")
        train_times = []
        test_times = []
        total_times = []
        load_times = []
        for epoch in range(args.epochs):
            start = time.perf_counter()
            load_time_train = train(epoch)
            train_time = time.perf_counter()
            load_time_test = test(epoch)
            test_time = time.perf_counter()

            train_times.append(train_time - start)
            test_times.append(test_time - train_time)
            total_times.append(test_time - start)
            load_times.append(load_time_train + load_time_test)
            scheduler.step()

        table = PrettyTable([])
        table.add_column("epoch", range(args.epochs))
        table.add_column("load time (for train + test)", load_times)
        table.add_column("train time", train_times)
        table.add_column("total_time", total_times)
        print(table, file=outfile)
        print(
            f"Average train time per epoch: {sum(train_times) / len(train_times)}",
            file=outfile,
        )
        print(
            f"Average loading time per epoch: {sum(load_times) / len(test_times)}",
            file=outfile,
        )
        print(
            f"Average total time per epoch: {sum(total_times) / len(total_times)}",
            file=outfile,
        )
        print(f"Total time for {args.epochs} epochs: ", sum(total_times), file=outfile)
        outfile.close()

    ###C3: I/O Optimization ###
    if args.exercise == "C3":
        outfile = open("C3.txt", "w")
        print("C3: I/O Optimization", file=outfile)

        num_workers = [0, 4, 8, 12]
        io_times = []

        for workers in num_workers:
            print(f"Number of workers: {workers}\n\n\n", file=outfile)
            args.num_workers = workers
            print("==> Preparing data..")
            trainloader, trainset, testloader, testset = load_data(args)
            runtime = 0
            for epoch in range(start_epoch, start_epoch + 200):
                load_time_train = train(epoch)
                load_time_test = test(epoch)
                scheduler.step()
                runtime += load_time_train + load_time_test
                print(f"Epoch {epoch} total loadtime: {runtime}", file=outfile)
            io_times.append(runtime)
            print(f"Num_workers {workers}- Total load time: {runtime}", file=outfile)

        table = PrettyTable([])
        table.add_column("num_workers", num_workers)
        table.add_column("io_times", io_times)
        print(table, file=outfile)

        outfile.close()


if __name__ == "__main__":
    main()
