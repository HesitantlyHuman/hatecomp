from typing import List, Tuple

import logging

import torch
from torch.utils.data import Dataset

class _HateDataset(Dataset):
    __name__ = 'None'
    downloader = None

    def __init__(
        self,
        path = None
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

    def _download(self, path: str):
        logging.info(f'Downloading {self.__name__} data to location f{path}.')
        downloader = self.downloader(
            save_path = path
        )
        downloader.load()

    def _load_data(self, path: str) -> Tuple[List]:
        return [], [], []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.ids[index], self.data[index], self.labels[index]