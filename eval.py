from common import parser
from pprint import pprint

import torch
from os.path import join
from torch.nn import NLLLoss
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

import pylab as pl
import time
import json

from tqdm import tqdm
from dotted.utils import dot
from copy import deepcopy

from helpful_files.networks import PROTO, avgpool, covapool, pL, pCL, fsL, fsCL, fbpredict, get_backbone
from helpful_files.testing import *


parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size in testing"
)

parser.add_argument(
    "--boxes_percentage",
    type=int,
    default=10,
    help="Percentage of images with bounding boxes available (few-shot localization models only)"
)

args = parser.parse_args()
args_dict = vars(args)
pprint(args_dict)


# Set Important Values

# General settings
datapath = str(args.data_root / args.dataset)           # The location of your train, test, repr, and query folders. 
experiment_path = args.results_root / args.dataset / args.experiment_name

model = str(experiment_path / args.weights_file)            # Where should your trained model(s) be saved, and under what name?
gpu = args.gpu_id                            # What gpu do you wish to train on?
workers = args.dl_workers                         # Number of cpu worker processes to use for data loading
verbosity = args.verbosity                      # How many batches in between status updates 
torch.cuda.set_device(gpu)
cudnn.benchmark = True

with (experiment_path / "args.json").open("r") as f:
    train_args = dot(json.load(f))
    pprint(train_args)
# Model construction
ensemble = train_args.n_ensembles                        # How many models to train in parallel
folding = train_args.bf                      # Use batch folding?
covariance_pooling = train_args.cp           # Use covariance pooling?
localizing = train_args.localizing                   # Use localization?
fewshot_local = train_args.fsl                # If you are using localization: few-shot, or parametric? Few-shot if True, param if False
network_width = train_args.network_width                 # Number of channels at every layer of the network
backbone = train_args.bkb

# Batch construction
bsize = args.batch_size                         # Batch size
boxes_available = 10                # Percentage of images with bounding boxes available (few-shot localization models only)
# k = args.k

# Data loading
include_masks = (localizing         # Include or ignore the bounding box annotations?
                 and fewshot_local)

n_trials = (10                      # Number of trials (few-shot localization models only)
            if include_masks else 1)


# Calculate embedding size based on model setup
d = (network_width if not 
     covariance_pooling else
     network_width**2)
if localizing and not covariance_pooling:
    d = network_width*2
assert n_trials == 1 or include_masks, ("Repeated trials will yield repeated identical results under this configuration."+
                                        "Please set ntrials to 1 or use a few-shot localizer.")

#########


experiment_path.mkdir(exist_ok=True, parents=True)
figs_path = experiment_path / "figures"
figs_path.mkdir(exist_ok=True)
dumpable_args = {
            k: v if isinstance(v, (int, str, bool, float)) else str(v)
            for (k, v) in args_dict.items()
        }


d_boxes = torch.load(join(datapath,'box_coords.pth'))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4905, 0.4961, 0.4330],std=[0.1737, 0.1713, 0.1779])
    ])

refr_dataset = datasets.ImageFolder(
    join(datapath,'refr'), 
    loader = lambda x: load_transform(x, d_boxes, transform, include_masks))
query_dataset = datasets.ImageFolder(
    join(datapath,'query'),
    loader = lambda x: load_transform(x, d_boxes, transform, include_masks))
refr_loader = torch.utils.data.DataLoader(
    refr_dataset, 
    batch_sampler = OrderedSampler(refr_dataset, bsize),
    num_workers = workers,
    pin_memory = True)
query_loader = torch.utils.data.DataLoader(
    query_dataset,
    batch_sampler = OrderedSampler(query_dataset, bsize),
    num_workers = workers,
    pin_memory = True)
way = len(refr_dataset.classes)

# Determine number of images with bounding boxes per-class
catsizes = torch.LongTensor(np.array([t[1] for t in refr_dataset.imgs])).bincount().float()
ngiv = (catsizes*boxes_available//100)
for i in range(ngiv.size(0)):
    if ngiv[i] == 0:
        ngiv[i] = 1
ngiv = ngiv.long().tolist()

print('Data loaded!')

models = [get_backbone(backbone, network_width).cuda() for i in range(ensemble)]
expander = avgpool()
if localizing:
    if fewshot_local:
        expander = fsCL if covariance_pooling else fsL
    else:
        expander = pCL(network_width) if covariance_pooling else pL(network_width)
elif covariance_pooling:
    expander = covapool
expanders = [deepcopy(expander) for _ in range(ensemble)]

# Load saved parameters
model_state = torch.load(model)
for i in range(ensemble):
    models[i].load_state_dict(model_state[i])
    models[i].eval()

# Load additional parameters for parametric localizer models
if localizing and not fewshot_local:
    fbcentroids = torch.load(model[:model.rfind('.')]+'_localizers'+model[model.rfind('.'):])
    for i in range(ensemble):
        expanders[i].centroids.data = fbcentroids[i]
        expanders[i].cuda()

print("Ready to go!")

acclists = {
    1: [],
    5: []
}
pcacclists = {
    1: [],
    5: []
}
alldispaccs = {
    1: np.zeros(way),
    5: np.zeros(way)
}
for r in tqdm(range(n_trials)):
    # Accumulate foreground/background prototypes, if using
    fbcentroids = (accumulateFB(models, refr_loader, way, network_width, ngiv, bsize)
                   if include_masks else 
                   [None]*ensemble)
    # Accumulate category prototypes
    centroids, counts = accumulate(models, refr_loader, expanders, 
                                   fbcentroids, way, d)
    # Score the models
    for k in [1, 5]:
        allacc, dispacc, perclassacc = score(k, centroids, fbcentroids, models, 
                                            query_loader, expanders, way)
        # Record statistics
        acclists[k] = acclists[k]+allacc
        pcacclists[k] = pcacclists[k]+list(perclassacc)
        alldispaccs[k] += dispacc

for k in [1,5]:

    acclist = acclists[k]
    pcacclist = pcacclists[k]
    alldispacc = alldispaccs[k]

    # Aggregate collected statistics
    accs = sum(acclist)/n_trials/ensemble
    pcaccs = sum(pcacclist)/n_trials/ensemble
    alldispacc = alldispacc/n_trials
    confs = 1.96*np.sqrt(np.var(acclist)/n_trials/ensemble)
    pcconfs = 1.96*np.sqrt(np.var(pcacclist)/n_trials/ensemble)

    # Report
    print(f"Top-{k}")
    print("Accuracies and 95% confidence intervals")
    print("Mean accuracy: \t\t%.2f \t+/- %.2f" % (accs*100, confs*100))
    print("Per-class accuracy: \t%.2f \t+/- %.2f" % (pcaccs*100, pcconfs*100))
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
                "n_trials": n_trials,
                "ensemble": ensemble
            },
            f,
            indent=2
        )
        
    logcounts = [np.log10(c) for c in counts]
    pl.figure()
    pl.axhline(0,color='k')
    pl.scatter(counts, dispacc*100, s=4)
    z = np.polyfit(logcounts, np.array(dispacc)*100, 1)
    p = np.poly1d(z)
    pl.plot([min(counts),max(counts)], [p(min(logcounts)),p(max(logcounts))], "r--")
    pl.ylim([0,100])
    pl.xlabel('# Reference Images')
    pl.ylabel('Percentage Points')
    pl.xscale('log')
    pl.title('Per-Class Top-%d Accuracy' % k)
    pl.savefig(experiment_path / f"test_top_{k}.png")


