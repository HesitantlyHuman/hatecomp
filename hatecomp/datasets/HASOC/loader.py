from typing import List, Tuple

import os
import csv
import logging

import torch
import numpy as np
from torch.utils.data import Dataset

from hatecomp.datasets.HASOC.download import HASOCDownloader, DEFAULT_DIRECTORY

DATA_RELATIVE_PATH = 'english_dataset'

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
        path = None,
        test = False,
        one_hot = True
    ):
        if path is None:
            save_path = DEFAULT_DIRECTORY
        data_dir = os.path.join(save_path, DATA_RELATIVE_PATH)

        self.save_path = save_path
        self.data_dir = data_dir
        try:
            self.ids, self.data, self.labels = self._load_data(self.data_dir, test = test)
        except FileNotFoundError:
            logging.info(f'HASOC data not found at expected location {self.data_dir}.')
            self._download(self.save_path)
            self.ids, self.data, self.labels = self._load_data(self.data_dir, test = test)
        self.one_hot = one_hot

    def _download(self, path: str):
        logging.info(f'Downloading HASOC data to location f{self.data_dir}.')
        downloader = HASOCDownloader(
            save_path = path
        )
        downloader.load()

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