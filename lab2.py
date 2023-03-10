"""Benchmark CIFAR10 with PyTorch on CPU & GPU"""
# Code Attribution: https://github.com/kuangliu/pytorch-cifar
import argparse
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from prettytable import PrettyTable

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
    parser.add_argument("--outfile", default="outfile.txt", help="output filename")
    parser.add_argument("--epochs", default=5, type=int, help="num epochs; default 5")
    parser.add_argument("--optimizer", default="SGD", help="optimizer, default SGD")
    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=4,
        help="dataloader workers; default 4",
    )
    parser.add_argument(
        "--data_path", default="./data", help="data dirpath; default ./data"
    )
    parser.add_argument(
        "--lr", default=0.1, type=float, help="learning rate, default 0.1"
    )
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="cuda usage; default False"
    )
    parser.add_argument(
        "--no_batch_norms",
        default=False,
        action="store_true",
        help="no batch norms; default False",
    )
    args = parser.parse_args()

    # Config
    print("==> Setting configs..")
    if args.exercise not in EXERCISES:
        raise ValueError("Invalid exercise")
    args.device = "cuda" if (torch.cuda.is_available() and args.cuda) else "cpu"
    args.optimizer = set_optimizer(args)
    args.filename = args.outfile if args.outfile else args.exercise + ".txt"
    print_config(args)
    outfile = open(args.filename, "a")

    # Data
    print("==> Preparing data..")
    trainloader, trainset, testloader, testset = load_data(args)

    # Model Setup
    print("==> Building model..")
    if args.no_batch_norms:
        net = ResNet18NoBatchNorm()
    else:
        net = ResNet18()
    net = net.to(args.device)
    if args.device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = args.optimizer(net.parameters(), **args.hyperparameters)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training
    def train(epoch):
        """
        Execute one epoch of training.
        Args:

            epoch (int): the current epoch
        Returns:
            c2_load_time (float): the time spent loading data
        """
        print("\nEpoch: %d" % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        sum_train_loss = 0

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
            sum_train_loss += train_loss
            c2_start = time.time()
        ave_train_loss = sum_train_loss / len(trainloader)
        return {"load_time": c2_load_time, "ave_train_loss": ave_train_loss}

    def test(epoch):
        """
        Tests the model on the test set.
        Args:
            epoch (int): the current epoch
        Returns:
            c2_load_time (float): the time spent loading data
        """
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

            return {"load_time": c2_load_time, "accuracy": 100 * correct / total}

    ###C2: Time Measurement###
    if args.exercise == "C2":
        print("======== C2: Time Measurement of Code ========\n\n", file=outfile)
        train_times = []
        test_times = []
        total_times = []
        load_times = []
        for epoch in range(args.epochs):
            start = time.perf_counter()
            load_time_train = train(epoch)["load_time"]
            train_time = time.perf_counter()
            load_time_test = test(epoch)["load_time"]
            test_time = time.perf_counter()

            train_times.append(train_time - start)
            test_times.append(test_time - train_time)
            total_times.append(test_time - start)
            load_times.append(load_time_train + load_time_test)
            scheduler.step()

        table = PrettyTable([])
        table.add_column("epoch", range(1, args.epochs + 1))
        table.add_column("data load time for train + test (secs)", load_times)
        table.add_column("train time (secs)", train_times)
        table.add_column("total time (secs)", total_times)
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
        print(f"Total time for {args.epochs} epochs: {sum(total_times)}", file=outfile)
        outfile.close()

    #### C3: I/O Optimization ####
    if args.exercise == "C3":
        print("======== C3: I/O Optimization ========\n\n", file=outfile)
        print(f"NUM_WORKERS {args.dataloader_workers} ####\n\n", file=outfile)

        print("==> Preparing data..")
        trainloader, trainset, testloader, testset = load_data(args)
        train_times = []
        test_times = []
        total_times = []
        load_times = []
        for epoch in range(args.epochs):
            start = time.perf_counter()
            load_time_train = train(epoch)["load_time"]
            train_time = time.perf_counter()
            load_time_test = test(epoch)["load_time"]
            test_time = time.perf_counter()

            train_times.append(train_time - start)
            test_times.append(test_time - train_time)
            total_times.append(test_time - start)
            load_times.append(load_time_train + load_time_test)
            scheduler.step()

            print(f"    Total load time: {sum(load_times)}")
            print(f"    Total compute time: {sum(total_times)}")

        print("#### C3 Summary ####\n\n", file=outfile)
        table = PrettyTable([])
        table.add_column("epoch", range(1, args.epochs + 1))
        table.add_column("data load time for train + test (secs)", load_times)
        table.add_column("train time (secs)", train_times)
        table.add_column("total time (secs)", total_times)
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
        print(f"Total time for {args.epochs} epochs: {sum(total_times)}", file=outfile)
        outfile.close()

    ####C4: Profiling Starting from C3####
    if args.exercise == "C4":
        print("======== C4: Profiling Starting from C3 ========\n\n", file=outfile)
        print(f"NUM_WORKERS {args.dataloader_workers} ####\n\n", file=outfile)

        print("==> Preparing data..")
        trainloader, trainset, testloader, testset = load_data(args)
        train_times = []
        test_times = []
        total_times = []
        load_times = []
        for epoch in range(args.epochs):
            start = time.perf_counter()
            load_time_train = train(epoch)["load_time"]
            train_time = time.perf_counter()
            load_time_test = test(epoch)["load_time"]
            test_time = time.perf_counter()

            train_times.append(train_time - start)
            test_times.append(test_time - train_time)
            total_times.append(test_time - start)
            load_times.append(load_time_train + load_time_test)
            scheduler.step()

            print(f"    Total load time: {sum(load_times)}")
            print(f"    Total compute time: {sum(total_times)}")

        print("#### C4 Summary ####\n\n", file=outfile)
        table = PrettyTable([])
        table.add_column("epoch", range(1, args.epochs + 1))
        table.add_column("data load time for train + test (secs)", load_times)
        table.add_column("train time (secs)", train_times)
        table.add_column("total time (secs)", total_times)
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
        print(f"Total time for {args.epochs} epochs: {sum(total_times)}", file=outfile)
        outfile.close()

    ####C5: Training GPU ####
    if args.exercise == "C5":
        print("======== C5: Training GPU ========\n\n", file=outfile)

        print(f"NUM_WORKERS {args.dataloader_workers} ####\n\n", file=outfile)

        print("==> Preparing data..")
        trainloader, trainset, testloader, testset = load_data(args)
        train_times = []
        test_times = []
        total_times = []
        load_times = []
        for epoch in range(args.epochs):
            start = time.perf_counter()
            load_time_train = train(epoch)["load_time"]
            train_time = time.perf_counter()
            load_time_test = test(epoch)["load_time"]
            test_time = time.perf_counter()

            train_times.append(train_time - start)
            test_times.append(test_time - train_time)
            total_times.append(test_time - start)
            load_times.append(load_time_train + load_time_test)
            scheduler.step()

            print(f"    Total load time: {sum(load_times)}")
            print(f"    Total compute time: {sum(total_times)}")

        print(f"#### C5 Summary ####\n\n", file=outfile)
        print(f"With GPU: Dataloading + Computation Time")
        table = PrettyTable([])
        table.add_column("epoch", range(1, args.epochs + 1))
        table.add_column("data load time for train + test (secs)", load_times)
        table.add_column("train time (secs)", train_times)
        table.add_column("total time (secs)", total_times)
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
        print(f"Total time for {args.epochs} epochs: {sum(total_times)}", file=outfile)
        outfile.close()

    #### C6: Experimenting with different optimizers ####
    if args.exercise == "C6":
        print(
            f"======== C6: Optimizer {args.optimizer_name} ========\n\n", file=outfile
        )
        args.dataloader_workers = 4
        args.device = "cuda"

        print("==> Preparing data..")
        trainloader, trainset, testloader, testset = load_data(args)

        print("==> Synchronizing GPU..")
        torch.cuda.synchronize()  # wait for warm-up to finish

        train_times = []
        accuracies = []
        average_train_losses = []
        for epoch in range(args.epochs):
            torch.cuda.synchronize()
            start_time = time.time()
            loss = train(epoch)["ave_train_loss"]
            train_time = time.time()
            scheduler.step()

            average_train_losses.append(loss)
            train_times.append(train_time - start_time)
            accuracy = test(epoch)["accuracy"]
            accuracies.append(accuracy)

            print(f"Epoch {epoch} ", file=outfile)
            print(f"    Train Time {train_time - start_time}\n", file=outfile)
            print(f"    Accuracy {accuracy}\n", file=outfile)

        print(
            f"#### C6 Summary For Optimizer {args.optimizer_name} ####\n\n",
            file=outfile,
        )
        table = PrettyTable([])
        table.add_column("Epoch", [i + 1 for i in range(args.epochs)])
        table.add_column("Training Time (secs)", train_times)
        table.add_column("Accuracy", accuracies)
        table.add_column("Average Train Loss", average_train_losses)
        print(table, file=outfile)
        outfile.close()

    #### C7: Experimenting with Batch Norm ####
    if args.exercise == "C7":
        print(f"======== C7: No Batch Norm ========\n\n", file=outfile)

        args.dataloader_workers = 4
        args.device = "cuda"
        print("==> Preparing data..")
        trainloader, trainset, testloader, testset = load_data(args)
        print("==> Synchronizing GPU..")
        torch.cuda.synchronize()  # wait for warm-up to finish

        train_times = []
        accuracies = []
        average_train_losses = []
        for epoch in range(args.epochs):
            torch.cuda.synchronize()  # wait for warm-up to finish

            start_time = time.time()
            loss = train(epoch)["ave_train_loss"]
            train_time = time.time()
            scheduler.step()

            average_train_losses.append(loss)
            train_times.append(train_time - start_time)
            accuracy = test(epoch)["accuracy"]
            accuracies.append(accuracy)

            print(f"Epoch {epoch} ", file=outfile)
            print(f"    Train Time {train_time - start_time}\n", file=outfile)
            print(f"    Accuracy {accuracy}\n", file=outfile)

        print(
            f"#### C6 Summary For Optimizer {args.optimizer_name} ####\n\n",
            file=outfile,
        )
        table = PrettyTable([])
        table.add_column("Epoch", [i + 1 for i in range(args.epochs)])
        table.add_column("Training Time (secs)", train_times)
        table.add_column("Accuracy", accuracies)
        table.add_column("Average Train Loss", average_train_losses)
        print(table, file=outfile)
        outfile.close()

    #### Q3: Num Trainable Parameters ####
    if args.exercise == "Q3":
        print(f"======== Q3: Num Trainable Parameters ========", file=outfile)
        print(f"Optimizer: {args.optimizer_name}\n", file=outfile)

        def count_parameters_and_gradients(model):
            table = PrettyTable(["Modules", "Trainable Parameters"])
            total_params = 0
            total_grads = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.grad():
                    total_grads += parameter.grad().numel()
                params = parameter.numel()
                table.add_row([name, params])
                total_params += params
            print(table, file=outfile)
            print(f"Total Trainable Params: {total_params}", file=outfile)
            print(f"Total Gradients: {total_grads}", file=outfile)

        count_parameters_and_gradients(net)
        return


if __name__ == "__main__":
    main()
