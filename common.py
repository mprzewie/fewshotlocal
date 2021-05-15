from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

parser = ArgumentParser(
    formatter_class=ArgumentDefaultsHelpFormatter,
    description="Few-shot local",
)

parser.add_argument(
    "--data_root",
    type=Path,
    default=Path("/mnt/users/mprzewiezlikowski/local_truten/uj/data/meta_inat/"),
    help="data root"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="meta_iNat",
    choices=["meta_iNat", "tiered_meta_iNat"],
    help="dataset"  
)

parser.add_argument(
    "--results_root",
    type=Path,
    default=Path("results"),
    help="results root"
)

parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Name of the experiment"
)

parser.add_argument(
    "--weights_file",
    type=str,
    default="weights.pth",
    help="weights file"
)

parser.add_argument(
    "--gpu_id",
    type=int,
    default=0,
    help="GPU id"
)

parser.add_argument(
    "--dl_workers",
    type=int,
    default=4,
    help="Number of cpu worker processes to use for data loading"
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
    "--verbosity",
    type=int,
    default=50,
    help="How many batches in between status updates"
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
