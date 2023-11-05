import argparse
import tkinter as tk
from tkinter.filedialog import askdirectory
import os
from pathlib import Path, PurePosixPath
from tqdm import tqdm
import time
import pandas as pd
from glob import glob
import json
from omegaconf import DictConfig

"""
Script to create an index of the primary directory of a dataset.
The index is saved as a list of paths in the dataset root.
"""


def run_cli():
    dataset_config_filename = "dataset_config.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", help="Dataset root directory", type=str)
    args = parser.parse_args()

    if args.dataset_root is not None:
        dataset_root = Path(args.dataset_root)
    else:
        tk.Tk().withdraw()
        if not (dataset_root := askdirectory(title="Select dataset root directory")):
            exit()
        dataset_root = Path(dataset_root)

    dataset_config_file = dataset_root / dataset_config_filename

    with open(dataset_config_file, "r") as f:
        dataset_config = DictConfig(json.load(f))

    print(f"Indexing directory: {str(dataset_root)}")
    directory_walk = index_directory(dataset_root / dataset_config.primary_directory)

    index_filename = dataset_config.directory_index_glob.replace("*.csv", "")
    index_path = dataset_root / f"{index_filename}_{time.strftime('%Y-%m-%dT%H_%M')}.csv"

    pd.DataFrame(directory_walk).to_csv(index_path, header=["dirpath", "dirnames", "files"], index=False)

    # Remove older indexes
    indexes = glob(str(dataset_root / dataset_config.directory_index_glob))
    to_remove = sorted(indexes)[:-1]
    for item in to_remove:
        os.remove(item)


def index_directory(directory, show_progress=True):
    directory_walk = []
    for i, (dirpath, dirnames, files) in tqdm(enumerate(os.walk(directory)), disable=not show_progress):
        directory_walk.append([PurePosixPath(Path(dirpath).relative_to(directory)), dirnames, files])

    return directory_walk


if __name__ == "__main__":
    run_cli()
