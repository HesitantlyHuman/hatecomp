from typing import Callable, List, Mapping, Tuple
import logging
import os

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from torch.utils.data.dataset import Subset

from hatecomp.base.utils import id_collate, tokenize_bookends, batch_and_slice

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class _HatecompDataset(IterableDataset):
    __name__ = "None"
    DOWNLOADER = None
    DEFAULT_DIRECTORY = None
    LABEL_KEY = None

    def __init__(
        self,
        root: str = None,
        download: bool = True,
    ):
        if root is None:
            if download:
                root = self.DEFAULT_DIRECTORY
            else:
                raise ValueError(
                    "root cannot be None if download is False. Either set root to the data root directory or download to True"
                )

        self.root = root
        try:
            self.ids, self.data, self.labels = self.load_data(self.root)
        except FileNotFoundError:
            logging.info(
                f"{self.__name__} data not found at expected location {self.root}."
            )
            if download:
                self.download(self.root)
                self.ids, self.data, self.labels = self.load_data(self.root)
            else:
                raise FileNotFoundError(f"Could not find data at {self.root}")
        assert (
            not self.LABEL_KEY is None
        ), f"{self.__name__} does not have a LABEL_KEY defined!"
        self.labels = self.encode_labels(self.LABEL_KEY)

    def split(self, p: float = 0.9) -> Tuple[Subset, Subset]:
        train_size = int(p * len(self))
        test_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, test_size])

    def download(self, path: str):
        logging.info(f"Downloading {self.__name__} data to location f{path}.")
        downloader = self.DOWNLOADER(save_path=path)
        downloader.load()

    def load_data(self, path: str) -> Tuple[list, list, list]:
        return list, list, list

    def encode_labels(self, encoding_scheme: dict) -> List:
        encoded_labels = []
        for labels in self.labels:
            encoded_labels.append(
                torch.squeeze(
                    torch.tensor([encoding_scheme[label] for label in labels])
                )
            )
        return torch.stack(encoded_labels, dim=0)

    def map(self, function: Callable, batched: bool = False, batch_size: int = 128):
        if not batched:
            batch_size = 1

        for (slice, data_group) in batch_and_slice(self.data, batch_size):
            mapped_data = function(data_group)
            assert len(mapped_data) == len(
                data_group
            ), "Mapping function did not return output of equal length over input batch!"
            self.data[slice] = mapped_data

        return self

    def split(self, test_proportion: float = 0.1):
        n_test = int(test_proportion * len(self))
        return torch.utils.data.random_split(self, [len(self) - n_test, n_test])

    def num_classes(self):
        return len(set(self.LABEL_KEY.values()))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        item = {"id": self.ids[index], "label": self.labels[index]}
        data = self.data[index]
        if isinstance(data, Mapping):
            item.update(data)
        else:
            item.update({"data": data})
        return item


class DataLoader(DataLoader):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.update({"collate_fn": id_collate})
        super().__init__(*args, **kwargs)
