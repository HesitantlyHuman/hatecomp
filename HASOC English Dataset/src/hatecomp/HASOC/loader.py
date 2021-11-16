from typing import List, Tuple

import os
import csv

import torch
import numpy as np
from torch.utils.data import Dataset

DOWNLOAD_DIR = './HASOC English Dataset/data'
DATA_RELATIVE_PATH = './english_dataset'

class HASOCDataset(Dataset):
    TSV_FILES = {
        'train' : 'english_dataset.tsv',
        'test' : 'hasoc2019_en_test-2919.tsv'
    }
    ENCODING_KEY = {
        'HOF' : 1,
        'NOT' : 0,
        'HATE' : 1,
        'OFFN' : 2,
        'PRFN' : 3,
        'TIN' : 1,
        'UNT' : 2,
        'NONE' : 0
    }

    def __init__(
        self,
        data_dir = None,
        test = False,
        one_hot = True
    ):
        if data_dir is None:
            data_dir = os.path.join(DOWNLOAD_DIR, DATA_RELATIVE_PATH)

        self.data_dir = data_dir
        self.ids, self.data, self.labels = self._load_data(self.data_dir, test = test)
        self.one_hot = one_hot

    def _load_data(self, path: str, test = False) -> Tuple[List]:
        if test:
            tsv_file = HASOCDataset.TSV_FILES['test']
        else:
            tsv_file = HASOCDataset.TSV_FILES['train']
        tsv_path = os.path.join(path, tsv_file)
        tsv = self._read_tsv(tsv_path)
        return self._convert_tsv(tsv)

    def _read_tsv(self, path: str) -> List[List[str]]:
        with open(path) as file:
            return list(csv.reader(file, delimiter = '\t'))

    def _convert_tsv(self, tsv: List[List[str]]) -> Tuple[List]:
        ids = []
        data = []
        labels = []
        for row in tsv[1:]:
            ids.append(row[0])
            data.append(row[1])
            labels.append([HASOCDataset.ENCODING_KEY[encoding] for encoding in row[2:]])
        return (np.array(data_list) for data_list in [ids, data, labels])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.ids[index], self.data[index], self.labels[index]

if __name__ == '__main__':
    dataset = HASOCDataset()
    print(len(dataset))