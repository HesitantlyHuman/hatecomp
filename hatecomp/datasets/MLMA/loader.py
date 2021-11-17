from typing import List, Tuple

import os
import csv
import logging

import torch
import numpy as np
from torch.utils.data import Dataset

from hatecomp.datasets.MLMA.download import MLMADownloader

class MLMADataset(Dataset):
    __name__ = 'MLMA'
    downloader = MLMADownloader

    def __init__(
        self,
        path = None,
        test = False,
        one_hot = True
    ):
        if path is None:
            save_path = self.downloader.DEFAULT_DIRECTORY

        self.save_path = save_path
        try:
            self.ids, self.data, self.labels = self._load_data(self.save_path, test = test)
        except FileNotFoundError:
            logging.info(f'{self.__name__} data not found at expected location {self.save_path}.')
            self._download(self.save_path)
            self.ids, self.data, self.labels = self._load_data(self.save_path, test = test)
        self.one_hot = one_hot

    def _download(self, path: str):
        logging.info(f'Downloading {self.__name__} data to location f{path}.')
        downloader = self.downloader(
            save_path = path
        )
        downloader.load()

    def _load_data(self, save_path: str, test = False) -> Tuple[List]:
        path = os.path.join(save_path, 'hate_speech_mlma/en_dataset_with_stop_words.csv')
        ids = []
        data = []
        labels = []
        with open(path) as file:
            for row in list(csv.reader(file))[1:]:
                ids.append(row[0])
                data.append(row[1])
                labels.append(row[2:])
        return (np.array(ls) for ls in [ids, data, labels])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.ids[index], self.data[index], self.labels[index]