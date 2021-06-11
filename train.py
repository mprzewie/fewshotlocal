import json
import sys
import time
from datetime import datetime
from os.path import join
from pprint import pprint

import pylab as pl
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import NLLLoss
from tqdm import tqdm

from common import parser
from helpful_files.networks import Network
from helpful_files.training import *

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
    "--n_ensembles",
    type=int,
    default=4,
    help="How many models to train in parallel"
)

parser.add_argument(
    "--way",
    type=int,
    default=20,
    help="Number of classes per batch during training"
)

parser.add_argument(
    "--train_shot",
    type=int,
    default=5,
    help="Number of images per class used to form prototypes"
)

parser.add_argument(
    "--test_shot",
    type=int,
    default=15,
    help="Number of images per class used to make predictions"
)

parser.add_argument(
    "--bf",
    action="store_true",
    default=False,
    help="Use batch folding? (true in notebook)"

)

parser.add_argument(
    "--cp",
    action="store_true",
    default=False,
    help="Use covariance pooling? (true in notebook)"

)

parser.add_argument(
    "--bkb",
    default="pn",
    choices=["pn", "resnet"],
    help="Backbone"
)

parser.add_argument(
    "--localizing",
    action="store_true",
    default=False,
    help="Use localization? (true in NB)"
)

parser.add_argument(
    "--fsl",
    action="store_true",
    default=False,
    help="If you are using localization: few-shot, or parametric? Few-shot if True, param if False"
)

parser.add_argument(
    "--augflip",
    action="store_true",
    default=False,
    help="Horizontal flip data augmentation (true in NB)"
)
parser.add_argument(
    "--network_width",
    type=int,
    default=64,
    help="Network width"
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="Learning rate"
)

args = parser.parse_args()
args_dict = vars(args)
pprint(args_dict)


# Set Important Values

# General settings
datapath = str(args.data_root / args.dataset)           # The location of your train, test, repr, and query folders. 
experiment_path = args.results_root / args.dataset / args.experiment_name
if args.without_symlinks:
    images_dir = args.images_dir
else:
    images_dir = None

savepath = str(experiment_path / args.weights_file)            # Where should your trained model(s) be saved, and under what name?
gpu = args.gpu_id                            # What gpu do you wish to train on?
workers = args.dl_workers                         # Number of cpu worker processes to use for data loading
epoch = args.n_epochs_in_lr_cut                          # Number of passes over the dataset before the learning rate is cut
ncuts = args.n_cuts                          # Number of times to cut the learning rate before training completes
verbosity = args.verbosity                      # How many batches in between status updates 
ensemble = args.n_ensembles                        # How many models to train in parallel
torch.cuda.set_device(gpu)
cudnn.benchmark = True

# Batch construction
way = args.way                            # Number of classes per batch during training
trainshot = args.train_shot                       # Number of images per class used to form prototypes
testshot = args.test_shot                       # Number of images per class used to make predictions
backbone = args.bkb

# Model construction
folding = args.bf                      # Use batch folding?
covariance_pooling = args.cp           # Use covariance pooling?
localizing = args.localizing                   # Use localization?
fewshot_local = args.fsl                # If you are using localization: few-shot, or parametric? Few-shot if True, param if False
network_width = args.network_width                 # Number of channels at every layer of the network
lr = args.lr
# Data loading
augmentation_flipping = args.augflip        # Horizontal flip data augmentation
include_masks = (localizing         # Include or ignore the bounding box annotations?
                 and fewshot_local)

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




d_boxes = torch.load(join(datapath,'box_coords.pth'))

transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4905, 0.4961, 0.4330],std=[0.1737, 0.1713, 0.1779])
    ])

if folding:
    # Batch folding has no reference/query distinction
    shots = [trainshot+testshot]
else:
    # Standard setup
    shots = [trainshot, testshot]
if localizing and fewshot_local and not folding:
    # Unfolded prototype localizers need another set of reference images to inform foreground/background predictions
    shots = [trainshot, trainshot, testshot-trainshot]
    
train_dataset = datasets.ImageFolder(
    join(datapath, 'train'),
    loader=lambda x: load_transform(x, d_boxes, transform, augmentation_flipping, include_masks, images_dir))
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_sampler = ProtoSampler(train_dataset, way, shots),
    num_workers = workers,
    pin_memory = True)
print('Data loaded!')

models = [Network(network_width, folding, covariance_pooling, 
                  localizing, fewshot_local, shots, backbone_kind=backbone).cuda() 
          for i in range(ensemble)]
optimizer = [optim.Adam(m.parameters(), lr=lr) for m in models]
scheduler = [optim.lr_scheduler.LambdaLR(o, lambda x: 1/(2**x)) for o in optimizer]
criterion = NLLLoss().cuda()

nweights = sum([i.numel() for i in list(models[0].parameters())])
print(nweights,"parameters in each neural net.")
print('Ready to go!')


start = time.time()
trainlosses, acctracker = [[] for _ in range(ensemble)],[[] for _ in range(ensemble)]
epochs = ncuts*epoch

for e in tqdm(range(epochs)):
    
    # Adjust learnrate
    if e%epoch == 0:
        [s.step() for s in scheduler]
    
    # Train for one epoch
    trainloss, acc = train(train_loader, models, optimizer, criterion, way, shots, verbosity)
    
    # Update the graphics, report
    # display.clear_output(wait=True)
    for j in range(ensemble):
        trainlosses[j].append(trainloss[j])
        acctracker[j].append(acc[j])
    pl.figure(1, figsize=(15,15))
    for i in range(ensemble):
        pl.subplot(ensemble,2,2*i+1)
        pl.plot(trainlosses[i])
        pl.ylim((0,3))
        pl.title(f"E {i}: Training Loss")
        pl.subplot(ensemble,2,2*i+2)
        pl.plot(acctracker[i])
        pl.ylim((0,1))
        pl.title(f"E {i}: Training Acc")
    pl.savefig(experiment_path / "training.png")
    with (experiment_path / "history.json").open("w") as f:
        json.dump({
            "args": dumpable_args,
            "train_loss": trainlosses,
            "train_acc": acctracker
        }, f)

    torch.save([m.encode.state_dict() for m in models], savepath)


    print("Training loss is: "+str(trainloss)+
            "\nTraining accuracy is: "+str(acc)+"\n")
    
        # If using parametric localization, save the extra parameters
    if localizing and not fewshot_local:
        torch.save([m.postprocess.centroids for m in models], 
                savepath[:savepath.rfind('.')]+'_localizers'+savepath[savepath.rfind('.'):])
    print("Models saved!")

    print("Approximately %.2f hours to completion"%(  (time.time()-start)/(e+1)*(epochs-e)/3600  ))
    print()
    
print("Training complete: %.2f hours total" % ((time.time()-start)/3600))
