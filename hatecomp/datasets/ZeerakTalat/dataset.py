from typing import List, Tuple

import os
import csv

import numpy as np
import torch

from hatecomp._path import install_path
from hatecomp.datasets.base.data import _HatecompDataset
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

    def encode_labels(self, encoding_scheme: dict) -> List:
        encoded_multiclass = []
        for labels in self.labels:
            encoded_labels = []
            expert, amateur = encoding_scheme[labels[0]], encoding_scheme[labels[1]]
            for labels in [expert, amateur]:
                if labels == 0:
                    encoded_labels += [0, 0]
                elif labels == 1:
                    encoded_labels += [1, 0]
                elif labels == 2:
                    encoded_labels += [0, 1]
                elif labels == 3:
                    encoded_labels += [1, 1]
            encoded_multiclass.append(torch.squeeze(torch.tensor(encoded_labels)))
        self.num_classes = [2, 2, 2, 2]
        return torch.stack(encoded_multiclass, dim=0)
