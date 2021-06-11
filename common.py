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
    choices=["meta_iNat", "tiered_meta_iNat", "cub"],
    help="dataset"  
)

parser.add_argument(
    "--without_symlinks",
    action="store_true",
    help='Only true when ran in system without symlinks, like Colab, Kaggle, etc.'
)

parser.add_argument(
    "--images_dir",
    type=Path,
    default=Path("../meta_inat/inat2017_84x84"),
    help="Path to images (only in runs without symlinks)"
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
    "--verbosity",
    type=int,
    default=50,
    help="How many batches in between status updates"
)


