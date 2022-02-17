from typing import List, Tuple

import os
import csv

import numpy as np
from torch.utils import data

from hatecomp._path import install_path
from hatecomp.base.data import _HatecompDataset
from hatecomp.datasets.ZeerakTalat.download import NAACLDownloader, NLPCSSDownloader


class NAACLDataset(_HatecompDataset):
    __name__ = "NAACL"
    DOWNLOADER = NAACLDownloader
    DEFAULT_DIRECTORY = os.path.join(install_path, "datasets/ZeerakTalat/data")

    CSV_FILE = "NAACL_SRW_2016.csv"
    LABEL_KEY = {"none": 0, "racism": 1, "sexism": 2}

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


class NLPCSSDataset(_HatecompDataset):
    __name__ = "NLPCSS"
    DOWNLOADER = NLPCSSDownloader
    DEFAULT_DIRECTORY = os.path.join(install_path, "datasets/ZeerakTalat/data")

    CSV_FILE = "NLP%2BCSS_2016.csv"
    LABEL_KEY = {"neither": 0, "link": 0, "racism": 1, "sexism": 2, "both": 3}

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
