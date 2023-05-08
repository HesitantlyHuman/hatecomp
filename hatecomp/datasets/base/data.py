from typing import Callable, List, Mapping, Tuple, Union, Sequence
import logging

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from torch.utils.data.dataset import Subset

from hatecomp.datasets.base.utils import id_collate, batch_and_slice


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

    def split(
        self, test_proportion: float = 0.1, class_minimum=None
    ) -> Tuple[Subset, Subset]:
        test_size = int(test_proportion * len(self))
        train_size = len(self) - test_size
        if not class_minimum is None:
            num_classes = self.num_classes
            all_classes_in_set = [False, False]
            while not all(all_classes_in_set):
                train_set, test_set = torch.utils.data.random_split(
                    self, [train_size, test_size]
                )
                for dataset_idx, dataset in enumerate([train_set, test_set]):
                    class_counts = [
                        np.zeros(class_count) for class_count in num_classes
                    ]
                    for item in dataset:
                        targets = item["label"]
                        for idx, target in enumerate(targets):
                            class_counts[idx][target] += 1
                    all_classes_in_set[dataset_idx] = all(
                        [
                            np.all(np.where(class_count > class_minimum, True, False))
                            for class_count in class_counts
                        ]
                    )
        else:
            train_set, test_set = torch.utils.data.random_split(
                self, [train_size, test_size]
            )
        return (train_set, test_set)

    def download(self, path: str):
        logging.info(f"Downloading {self.__name__} data to location f{path}.")
        downloader = self.DOWNLOADER(save_path=path)
        downloader.load()

    def load_data(self, path: str) -> Tuple[list, list, list]:
        raise NotImplementedError

    def encode_labels(self, encoding_scheme: dict) -> List:
        class_values = [set() for i in self.labels[0]]
        encoded_labels = []
        for labels in self.labels:
            encoded_example = []
            for idx, label in enumerate(labels):
                encoded_label = encoding_scheme[label]
                class_values[idx].add(encoded_label)
                encoded_example.append(encoded_label)
            encoded_labels.append(torch.squeeze(torch.tensor(encoded_example)))
        self.num_classes = [len(class_options) for class_options in class_values]
        return encoded_labels

    def map(self, function: Callable, batched: bool = False, batch_size: int = 128):
        if not batched:
            batch_size = 1

        for slice, data_group in batch_and_slice(self.data, batch_size):
            mapped_data = function(data_group)
            if isinstance(mapped_data, Sequence):
                assert len(mapped_data) == len(
                    data_group
                ), "Mapping function did not return output of equal length over input batch!"
            else:
                mapped_data = [mapped_data]
            self.data[slice] = mapped_data

        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Union[int, slice]) -> Tuple:
        if isinstance(index, slice):
            dataset_class = type(self)
            dataset_view = dataset_class.__new__(dataset_class)
            dataset_view.ids = self.ids[index]
            dataset_view.data = self.data[index]
            dataset_view.labels = self.labels[index]
            return dataset_view
        else:
            item = {"id": self.ids[index], "label": self.labels[index]}
            data = self.data[index]
            if isinstance(data, Mapping):
                item.update(data)
            else:
                item.update({"data": data})
            return item

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class DataLoader(DataLoader):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.update({"collate_fn": id_collate})
        super().__init__(*args, **kwargs)
