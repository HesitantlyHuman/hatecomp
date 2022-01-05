from typing import List, Tuple

import os
import csv

import numpy as np
from torch.utils import data

from hatecomp._path import install_path
from hatecomp.base.dataset import _HatecompDataset
from hatecomp.datasets.TwitterSexism.download import TwitterSexismDownloader

class TwitterSexismDataset(_HatecompDataset):
    __name__ = 'TwitterSexism'
    DOWNLOADER = TwitterSexismDownloader
    DEFAULT_DIRECTORY = os.path.join(install_path, 'datasets/TwitterSexism/data')

    CSV_FILE = 'twitter_sexism.csv'
    ENCODING_KEY = {
        'none' : 0,
        'benevolent' : 1,
        'hostile' : 2
    }

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
        
if __name__ == '__main__':
    dataset = TwitterSexismDataset(download = True)
    print(dataset[0])
    print(len(dataset))