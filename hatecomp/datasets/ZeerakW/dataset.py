from typing import List, Tuple

import os
import csv

import numpy as np
from torch.utils import data

from hatecomp._path import install_path
from hatecomp.base.data import _HatecompDataset
from hatecomp.datasets.ZeerakW.download import NAACLDownloader, NLPCSSDownloader


class NAACLDataset(_HatecompDataset):
    __name__ = "NAACL"
    DOWNLOADER = NAACLDownloader
    DEFAULT_DIRECTORY = os.path.join(install_path, "datasets/ZeerakW/data")

    CSV_FILE = "NAACL_SRW_2016.csv"
    ENCODING_KEY = {"none": 0, "racism": 1, "sexism": 2}

    def __init__(self, root: str = None, download=False):
        super().__init__(root=root, download=download)

    def load_data(self, path: str) -> Tuple[List]:
        with open(os.path.join(path, self.CSV_FILE)) as data_file:
            csv_data = list(csv.reader(data_file))
        ids, data, labels = [], [], []
        for item in csv_data[1:]:
            ids.append(item[0])
            data.append(item[1])
            labels.append(item[2:])
        return (np.array(ids), np.array(data), np.array(labels))


class NLPCSSDataset(_HatecompDataset):
    __name__ = "NLPCSS"
    DOWNLOADER = NLPCSSDownloader
    DEFAULT_DIRECTORY = os.path.join(install_path, "datasets/ZeerakW/data")

    CSV_FILE = "NLP%2BCSS_2016.csv"
    ENCODING_KEY = {"none": 0, "racism": 1, "sexism": 2}

    def __init__(self, root: str = None, download=False):
        super().__init__(root=root, download=download)

    def _load_data(self, path: str) -> Tuple[List]:
        with open(os.path.join(path, self.CSV_FILE)) as data_file:
            csv_data = list(csv.reader(data_file))
        ids, data, labels = [], [], []
        for item in csv_data[1:]:
            ids.append(item[0])
            data.append(item[1])
            labels.append(item[2:])
        return (np.array(ids), np.array(data), np.array(labels))


if __name__ == "__main__":
    dataset = NAACLDataset(download=True)
    train, test = dataset.split()
    print(train[0])
    print(test[0])
    dataset = NLPCSSDataset(download=True)
    train, test = dataset.split()
    print(train[0])
    print(test[0])
