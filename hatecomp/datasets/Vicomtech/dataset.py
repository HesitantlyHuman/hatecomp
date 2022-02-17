from typing import List, Tuple

import os
import csv

import numpy as np

from hatecomp.base.data import _HatecompDataset
from hatecomp._path import install_path
from hatecomp.datasets.Vicomtech.download import VicomtechDownloader


class VicomtechDataset(_HatecompDataset):
    __name__ = "Vicomtech"
    DOWNLOADER = VicomtechDownloader
    DEFAULT_DIRECTORY = os.path.join(install_path, "datasets/Vicomtech/data")

    CSV_FILE = "vicomtech.csv"
    LABEL_KEY = {
        "noHate": 0,
        "hate": 1,
    }

    def __init__(self, root: str = None, download: bool = True):
        super().__init__(root=root, download=download)

    def load_data(self, path: str) -> Tuple[List]:
        with open(os.path.join(path, self.CSV_FILE)) as data_file:
            csv_data = list(csv.reader(data_file))
        ids, data, labels = [], [], []
        for item in csv_data[1:]:
            ids.append(item[0])
            data.append(item[1])
            labels.append(item[2:])
        return (np.array(ids), data, np.array(labels))


if __name__ == "__main__":
    dataset = VicomtechDataset(download=True)
    print(dataset[0])
