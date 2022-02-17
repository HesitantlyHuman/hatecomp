from typing import List, Tuple

import os
import csv

import numpy as np
from torch.utils import data

from hatecomp._path import install_path
from hatecomp.base.data import _HatecompDataset
from hatecomp.datasets.TwitterSexism.download import TwitterSexismDownloader


class TwitterSexismDataset(_HatecompDataset):
    __name__ = "TwitterSexism"
    DOWNLOADER = TwitterSexismDownloader
    DEFAULT_DIRECTORY = os.path.join(install_path, "datasets/TwitterSexism/data")

    CSV_FILE = "twitter_sexism.csv"
    LABEL_KEY = {"benevolent": 0, "hostile": 1}

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
