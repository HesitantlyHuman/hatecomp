from typing import List, Tuple

import logging

import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataset import Subset

class _HatecompDataset(Dataset):
    __name__ = 'None'
    DOWNLOADER = None

    def __init__(
        self,
        root: str = None,
        download = False,
    ):
        if root is None and download is False:
            raise ValueError('root cannot be None if download is False. Either set root to the data root directory or download to True')
        
        self.root = root
        try:
            self.ids, self.data, self.labels = self._load_data(self.root)
        except FileNotFoundError:
            logging.info(f'{self.__name__} data not found at expected location {self.root}.')
            if download:
                self._download(self.root)
                self.ids, self.data, self.labels = self._load_data(self.root)
            else:
                raise FileNotFoundError(f'Could not find data at {self.root}')

    def split(self, p = 0.9) -> Tuple[Subset, Subset]:
        train_size = int(p * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])

    def _download(self, path: str):
        logging.info(f'Downloading {self.__name__} data to location f{path}.')
        downloader = self.downloader(
            save_path = path
        )
        downloader.load()

    def _load_data(self, path: str) -> Tuple[np.array, np.array, np.array]:
        return np.array([]), np.array([]), np.array([])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        return self.ids[index], self.data[index], self.labels[index]