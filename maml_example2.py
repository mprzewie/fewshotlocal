
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
import os
from tqdm import tqdm
from pathlib import Path
import dill

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()

def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def main(datapath, lr=0.005, maml_lr=0.01, iterations=1000, ways=5, shots=1, meta_batch_size=32, fas=5, device=torch.device("cuda"),
         download_location='~/data'):

    TRAIN = "train"
    VAL = "refr"
    TEST = "query"

    data_transforms = {
        TRAIN: transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4905, 0.4961, 0.4330],std=[0.1737, 0.1713, 0.1779])
        ]),
        TEST: transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4905, 0.4961, 0.4330],std=[0.1737, 0.1713, 0.1779])
        ]),
        VAL: transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4905, 0.4961, 0.4330],std=[0.1737, 0.1713, 0.1779])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(datapath, x), data_transforms[x]) for x in [TRAIN, VAL, TEST]}
    class_names = image_datasets[TRAIN].classes
    meta_path = Path(datapath) / "meta_datasets.pth"

    if meta_path.exists():
        dataset_dict = torch.load(meta_path)
    else:

        dataset_dict = {x: l2l.data.MetaDataset(image_datasets[x]) for x in tqdm([TRAIN, VAL, TEST], "Meta datasets")}
        torch.save(dataset_dict, meta_path, pickle_module=dill)


    train_dataset = dataset_dict[TRAIN]
    valid_dataset = dataset_dict[VAL]
    test_dataset = dataset_dict[TEST]

    train_transforms = [l2l.data.transforms.NWays(dataset_dict[TRAIN], ways),
                  l2l.data.transforms.KShots(dataset_dict[TRAIN], 2*shots),
                  l2l.data.transforms.LoadData(dataset_dict[TRAIN])
                  ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)
                                       
    valid_transforms = [ l2l.data.transforms.NWays(dataset_dict[VAL], ways),
                  l2l.data.transforms.KShots(dataset_dict[VAL], 2*shots),
                  l2l.data.transforms.LoadData(dataset_dict[VAL])
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=600)

    test_transforms = [
        l2l.data.transforms.NWays(dataset_dict[TEST], ways),
                  l2l.data.transforms.KShots(dataset_dict[TEST], 2*shots),
                  l2l.data.transforms.LoadData(dataset_dict[TEST])
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=600)

    print("Data loaded!")
    

    model = l2l.vision.models.CNN4(len(class_names))

    model.to(device)
    print(model)
    maml = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(maml.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    adaptation_steps = 1

    for iteration in range(iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for _ in range(meta_batch_size):
            learner = maml.clone()
            train_task = train_tasks.sample()
            batch = train_task
            # data, labels = train_task
            # data = data[0].to(device)
            # labels = labels.to(device)

            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy

            # Compute meta-validation loss
            learner = maml.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy
        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = test_tasks.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn2Learn MNIST Example')

    parser.add_argument('--datapath', type=str, help='Path to data')
    parser.add_argument('--ways', type=int, default=5, metavar='N',
                        help='number of ways (default: 5)')
    parser.add_argument('--shots', type=int, default=1, metavar='N',
                        help='number of shots (default: 1)')
    parser.add_argument('-meta_batch_size', '--tasks-per-step', type=int, default=32, metavar='N',
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
         meta_batch_size=args.tasks_per_step,
         fas=args.fast_adaption_steps,
         device=device,
         download_location=args.download_location)
