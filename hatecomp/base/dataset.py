from typing import Callable, List, Tuple

import logging
import os

import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataset import Subset

from hatecomp.base.utils import tokenize_bookends

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class TokenizerDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: Callable) -> None:
        super().__init__()

        self.data = self.tokenize_data(dataset, tokenizer)

    def tokenize_data(self, dataset: Dataset, tokenizer: Callable) -> List:
        return [self.tokenize_single(item, tokenizer) for item in dataset.data]

    def tokenize_single(self, item: dict, tokenizer: Callable) -> dict:
        formatted_item = {
            'id' : item['id'],
            'label' : torch.tensor(item['label'])
        }
        formatted_item.update({key: torch.tensor(value) for key, value in tokenizer(str(item['data'])).items()})
        return formatted_item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        return {key : value for key, value in data_item.items() if not key == 'id'}

    def from_huggingface(dataset: Dataset, tokenizer: Callable, max_token_length = 512):
        tokenizer_function = lambda string_in : tokenize_bookends(
            string_in,
            max_token_length,
            lambda x : tokenizer(x, padding = 'max_length', max_length = max_token_length)
        )
        return TokenizerDataset(dataset, tokenizer_function)

class _HatecompDataset(Dataset):
    __name__ = 'None'
    DOWNLOADER = None
    DEFAULT_DIRECTORY = None
    LABEL_ENCODING = None

    def __init__(
        self,
        root: str = None,
        download = False,
    ):
        if root is None:
            if download:
                root = self.DEFAULT_DIRECTORY
            else:
                raise ValueError('root cannot be None if download is False. Either set root to the data root directory or download to True')
        
        self.root = root
        try:
            self.ids, self.data, self.labels = self.load_data(self.root)
        except FileNotFoundError:
            logging.info(f'{self.__name__} data not found at expected location {self.root}.')
            if download:
                self.download(self.root)
                self.ids, self.data, self.labels = self.load_data(self.root)
            else:
                raise FileNotFoundError(f'Could not find data at {self.root}')
        self.labels = self.encode_labels(self.LABEL_ENCODING)

    def split(self, p = 0.9) -> Tuple[Subset, Subset]:
        train_size = int(p * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])

    def download(self, path: str):
        logging.info(f'Downloading {self.__name__} data to location f{path}.')
        downloader = self.DOWNLOADER(
            save_path = path
        )
        downloader.load()

    def load_data(self, path: str) -> Tuple[np.array, np.array, np.array]:
        return np.array([]), np.array([]), np.array([])

    def encode_labels(self, encoding_scheme: dict) -> List:
        encoded_labels = []
        for labels in self.labels:
            encoded_labels.append([encoding_scheme[label] for label in labels])
        return encoded_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        return {
            'id' : self.ids[index], 
            'data' : self.data[index],
            'label' : self.labels[index]
        }