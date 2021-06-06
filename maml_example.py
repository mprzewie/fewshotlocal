#!/usr/bin/env python3

import argparse
import random

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.datasets import MNIST
from os.path import join
from pathlib import Path
import learn2learn as l2l
from helpful_files.training import *


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def main(datapath, lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, tps=32, fas=5, device=torch.device("cuda"),
         download_location='~/data'):
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x.view(1, 28, 28),
    ])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4905, 0.4961, 0.4330], std=[0.1737, 0.1713, 0.1779])
    ])

    train_dataset = datasets.ImageFolder(
        join(datapath, 'train'),
        loader=lambda x: load_transform(x, None, transform, False, False))
    print(type(train_dataset))
    print('Data loaded!')

    iNat_train = l2l.data.MetaDataset(train_dataset)
    train_tasks = l2l.data.TaskDataset(iNat_train,
                                       task_transforms=[
                                           l2l.data.transforms.NWays(iNat_train, ways),
                                           l2l.data.transforms.KShots(iNat_train, 2 * shots),
                                           l2l.data.transforms.LoadData(iNat_train),
                                           l2l.data.transforms.RemapLabels(iNat_train),
                                           l2l.data.transforms.ConsecutiveLabels(iNat_train),
                                       ],
                                       num_tasks=1000)

    model = l2l.vision.models.CNN4(ways)
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.NLLLoss(reduction='mean')

    for iteration in range(iterations):
        iteration_error = 0.0
        iteration_acc = 0.0
        for _ in range(tps):
            learner = meta_model.clone()
            train_task = train_tasks.sample()
            data, labels = train_task
            data = data[0].to(device)
            labels = labels.to(device)

            # Separate data into adaptation/evalutation sets
            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            adaptation_indices[np.arange(shots * ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            # Fast Adaptation
            for step in range(fas):
                train_error = loss_func(learner(adaptation_data), adaptation_labels)
                learner.adapt(train_error)

            # Compute validation loss
            predictions = learner(evaluation_data)
            valid_error = loss_func(predictions, evaluation_labels)
            valid_error /= len(evaluation_data)
            valid_accuracy = accuracy(predictions, evaluation_labels)
            iteration_error += valid_error
            iteration_acc += valid_accuracy

        iteration_error /= tps
        iteration_acc /= tps
        print('Loss : {:.3f} Acc : {:.3f}'.format(iteration_error.item(), iteration_acc))

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument('--datapath', type=str, help='Path to data')
    parser.add_argument('--ways', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=1, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('-tps', '--tasks-per-step', type=int, default=32, metavar='N',
                        help='tasks per step (default: 32)')
    parser.add_argument('-fas', '--fast-adaption-steps', type=int, default=5, metavar='N',
                        help='steps per fast adaption (default: 5)')

    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations (default: 1000)')

    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--maml-lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for MAML (default: 0.01)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--download-location', type=str, default="/tmp/mnist", metavar='S',
                        help='download location for train data (default : /tmp/mnist')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    main(datapath=args.datapath,
         lr=args.lr,
         maml_lr=args.maml_lr,
         iterations=args.iterations,
         ways=args.ways,
         shots=args.shots,
         tps=args.tasks_per_step,
         fas=args.fast_adaption_steps,
         device=device,
         download_location=args.download_location)
