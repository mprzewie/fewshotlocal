import json
import sys
import time
from datetime import datetime
from os.path import join
from pprint import pprint

import pylab as pl
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import NLLLoss
from torchvision.models import resnet50
from tqdm import tqdm
from pathlib import Path
from common import parser
from helpful_files.training import *
from helpful_files import testing as tst

parser.add_argument(
    "--n_ensembles",
    type=int,
    default=1,
    help="How many models to train in parallel"
)

parser.add_argument(
    "--augflip",
    action="store_true",
    default=True,
    help="Horizontal flip data augmentation (true in NB)"
)

parser.add_argument(
    "--n_epochs_in_lr_cut",
    type=int,
    default=10,
    help="Number of passes over the dataset before the learning rate is cut"
)

parser.add_argument(
    "--n_cuts",
    type=int,
    default=5,
    help="Number of times to cut the learning rate before training completes"
)

parser.add_argument(
    "--ft_epochs",
    type=int,
    default=10,
    help="Number of epochs for fine-tuning"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size"
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    help="LR"
)

parser.add_argument(
    "--ft_lr",
    type=float,
    default=0.0001,
    help="LR (Fine-tuning)"
)

parser.add_argument(
    "--ft_freeze",
    action="store_true",
    help="Freeze ResNet when finetuning"

)


parser.add_argument(
    "--ctd",
    action="store_true",
    help="Load model from saved weights befroe pretraining"
)




args = parser.parse_args()
args_dict = vars(args)
pprint(args_dict)

# Set Important Values

# General settings
datapath = str(args.data_root / args.dataset)  # The location of your train, test, repr, and query folders.
experiment_path = args.results_root / args.dataset / args.experiment_name

savepath = str(experiment_path / args.weights_file)  # Where should your trained model(s) be saved, and under what name?
gpu = args.gpu_id  # What gpu do you wish to train on?
epoch = args.n_epochs_in_lr_cut  # Number of passes over the dataset before the learning rate is cut
ncuts = args.n_cuts  # Number of times to cut the learning rate before training completes
workers = args.dl_workers  # Number of cpu worker processes to use for data loading
verbosity = args.verbosity  # How many batches in between status updates
ensemble = args.n_ensembles  # How many models to train in parallel
torch.cuda.set_device(gpu)
cudnn.benchmark = True

lr = args.lr
ft_lr = args.ft_lr
# Data loading
augmentation_flipping = args.augflip  # Horizontal flip data augmentation
include_masks = False

experiment_path.mkdir(exist_ok=True, parents=True)
dumpable_args = {
    k: v if isinstance(v, (int, str, bool, float)) else str(v)
    for (k, v) in args_dict.items()
}
with (experiment_path / "args.json").open("w") as f:
    json.dump(
        dumpable_args,
        f,
        indent=2,
    )

with (experiment_path / "rerun.sh").open("w") as f:
    print("#", datetime.now(), file=f)
    print("python", *sys.argv, file=f)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.4905, 0.4961, 0.4330], std=[0.1737, 0.1713, 0.1779])
])

train_dataset = datasets.ImageFolder(
    join(datapath, 'train'),
    loader=lambda x: load_transform(x, None, transform, augmentation_flipping, include_masks)
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=True)

refr_dataset = datasets.ImageFolder(
    join(datapath, 'refr'),
    loader=lambda x: load_transform(x, None, transform, False, False)
)
query_dataset = datasets.ImageFolder(
    join(datapath, 'query'),
    loader=lambda x: load_transform(x, None, transform, False, False)
)
refr_loader = torch.utils.data.DataLoader(
    refr_dataset,
    num_workers=workers,
    batch_sampler = tst.OrderedSampler(refr_dataset, args.batch_size),
    pin_memory=True)
query_loader = torch.utils.data.DataLoader(
    query_dataset,
    batch_sampler=tst.OrderedSampler(query_dataset, args.batch_size),
    num_workers=workers,
    pin_memory=True)

print('Data loaded!')

models = [resnet50(pretrained=True) for _ in range(ensemble)]
for model in models:
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model.cuda()
    model.train()

if Path(savepath).exists() and args.ctd:
    print("Attempting to load weights from", savepath)
    weights = torch.load(savepath, map_location="cpu")
    for m, sd in zip(models, weights):
        m.load_state_dict(sd)
        m.cuda()
        m.train()

optimizer = [optim.Adam(m.parameters(), lr=lr) for m in models]
scheduler = [optim.lr_scheduler.LambdaLR(o, lambda x: 1 / (2 ** x)) for o in optimizer]
criterion = nn.CrossEntropyLoss()

nweights = sum([i.numel() for i in list(models[0].parameters())])
print(nweights, "parameters in each neural net.")
print('Ready to go!')

start = time.time()
trainlosses, acctracker = [[] for _ in range(ensemble)], [[] for _ in range(ensemble)]
epochs = ncuts * epoch
test_way = len(refr_dataset.classes)

for e in tqdm(range(epochs)):

    # Adjust learnrate
    if e % epoch == 0:
        [s.step() for s in scheduler]

    # Train for one epoch
    trainloss, acc = train_resnet(train_loader, models, optimizer, criterion, verbosity)

    # Update the graphics, report
    # display.clear_output(wait=True)
    for j in range(ensemble):
        trainlosses[j].append(trainloss[j])
        acctracker[j].append(acc[j])
    pl.figure(1, figsize=(15, 15))
    for i in range(ensemble):
        pl.subplot(ensemble, 2, 2 * i + 1)
        pl.plot(trainlosses[i])
        pl.ylim((0, 3))
        pl.title(f"E {i}: Training Loss")
        pl.subplot(ensemble, 2, 2 * i + 2)
        pl.plot(acctracker[i])
        pl.ylim((0, 1))
        pl.title(f"E {i}: Training Acc")
    pl.savefig(experiment_path / "training.png")
    with (experiment_path / "history.json").open("w") as f:
        json.dump({
            "args": dumpable_args,
            "train_loss": trainlosses,
            "train_acc": acctracker
        }, f)

    torch.save([m.state_dict() for m in models], savepath)

    print("Training loss is: " + str(trainloss) +
          "\nTraining accuracy is: " + str(acc) + "\n")

    print("Models saved!")

    print("Approximately %.2f hours to completion" % ((time.time() - start) / (e + 1) * (epochs - e) / 3600))
    print()

print("Training complete: %.2f hours total" % ((time.time() - start) / 3600))

print('Fine-tuning on refr dataset')
optimizer = [optim.Adam(m.parameters(), lr=ft_lr) for m in models]
for i in range(ensemble):
    if args.ft_freeze:
        for param in models[i].parameters():
            param.requires_grad = False
    models[i].fc = nn.Linear(models[i].fc.in_features, test_way)
    models[i].cuda()

start = time.time()

for e in tqdm(range(args.ft_epochs)):
    # Train for one epoch
    trainloss, acc = train_resnet(refr_loader, models, optimizer, criterion, verbosity)

    print("Fine-tune loss is: " + str(trainloss) +
          "\nFine-tune accuracy is: " + str(acc) + "\n")

    # Update the graphics, report
    # display.clear_output(wait=True)
    for j in range(ensemble):
        trainlosses[j].append(trainloss[j])
        acctracker[j].append(acc[j])
    pl.figure(1, figsize=(15, 15))
    for i in range(ensemble):
        pl.subplot(ensemble, 2, 2 * i + 1)
        pl.plot(trainlosses[i])
        pl.ylim((0, 3))
        pl.title(f"E {i}: Fine-tuning Loss")
        pl.subplot(ensemble, 2, 2 * i + 2)
        pl.plot(acctracker[i])
        pl.ylim((0, 1))
        pl.title(f"E {i}: Fine-tuning Acc")
    pl.savefig(experiment_path / "fine-tuning.png")
    with (experiment_path / "history-ft.json").open("w") as f:
        json.dump({
            "args": dumpable_args,
            "ft_loss": trainlosses,
            "ft_acc": acctracker
        }, f)

    # torch.save([m.state_dict() for m in models], savepath)

    print("Approximately %.2f hours to completion" % ((time.time() - start) / (e + 1) * (args.ft_epochs - e) / 3600))
    print()

acclists = {
    1: [],
    5: []
}
pcacclists = {
    1: [],
    5: []
}
alldispaccs = {
    1: np.zeros(test_way),
    5: np.zeros(test_way)
}

for model in models:
    model.eval()

# Score the models
for k in [1, 5]:
    allacc, dispacc, perclassacc = tst.score_resnet(k, models, query_loader, test_way)
    # Record statistics
    acclists[k] = acclists[k] + allacc
    pcacclists[k] = pcacclists[k] + list(perclassacc)
    alldispaccs[k] += dispacc

for k in [1, 5]:
    acclist = acclists[k]
    pcacclist = pcacclists[k]
    alldispacc = alldispaccs[k]

    # Aggregate collected statistics
    accs = sum(acclist) / ensemble
    pcaccs = sum(pcacclist) / ensemble
    alldispacc = alldispacc
    confs = 1.96 * np.sqrt(np.var(acclist) / ensemble)
    pcconfs = 1.96 * np.sqrt(np.var(pcacclist) / ensemble)

    # Report
    print(f"Top-{k}")
    print("Accuracies and 95% confidence intervals")
    print("Mean accuracy: \t\t%.2f \t+/- %.2f" % (accs * 100, confs * 100))
    print("Per-class accuracy: \t%.2f \t+/- %.2f" % (pcaccs * 100, pcconfs * 100))
    print("=====")
    with (experiment_path / f"test_top_{k}.json").open("w") as f:
        json.dump(
            {
                "top_k": k,
                "accuracy": {
                    "mean": accs,
                    "std": confs,
                    "list": acclist
                },
                "per_class_accuracy": {
                    "mean": pcaccs,
                    "std": pcconfs,
                    "list": pcacclist
                },
                "ensemble": ensemble
            },
            f,
            indent=2
        )
