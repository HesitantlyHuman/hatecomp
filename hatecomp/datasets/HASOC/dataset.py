from typing import List, Tuple

import os
import csv

import numpy as np

from hatecomp.base.data import _HatecompDataset
from hatecomp._path import install_path
from hatecomp.datasets.HASOC.download import HASOCDownloader


class HASOCDataset(_HatecompDataset):
    __name__ = "HASOC"
    DOWNLOADER = HASOCDownloader
    DEFAULT_DIRECTORY = DEFAULT_DIRECTORY = os.path.join(
        install_path, "datasets/HASOC/data"
    )
    LABEL_KEY = {
        "HOF": 1,
        "NOT": 0,
        "HATE": 1,
        "OFFN": 2,
        "PRFN": 3,
        "TIN": 1,
        "UNT": 2,
        "NONE": 0,
    }

    CSV_FILE = "hasoc.csv"

    def __init__(self, root: str = None, download=True):
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
    dataset = HASOCDataset(download=True)
    print(dataset[600])
