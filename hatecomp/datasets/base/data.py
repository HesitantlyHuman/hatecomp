from typing import Callable, Tuple, List, Union
import logging
import random
import os
from torch.utils.data import Dataset
import hatecomp
from hatecomp.datasets.base.utils import batch_enumerate, map_functions, get_unique


class FunctionalDataset(Dataset):
    def __init__(self) -> None:
        self.data_transforms = []
        self.target_transforms = []
        self.has_data_transforms = False
        self.has_target_transforms = False

    def split(
        self, test_proportion: float = 0.1, seed: int = None
    ) -> Tuple["HatecompDatasetView", "HatecompDatasetView"]:
        dataset_length = len(self)
        indices = list(range(dataset_length))
        random.Random(seed).shuffle(indices)
        n_test = int(dataset_length * test_proportion)
        return (
            HatecompDatasetView(self, indices[n_test:], sorted=False),
            HatecompDatasetView(self, indices[:n_test], sorted=False),
        )

    def map(
        self,
        function: Union[Callable, List[Callable]],
        targets: bool = False,
        batch_size: int = None,
    ) -> Union["HatecompDataset", "HatecompDatasetView"]:
        raise NotImplementedError

    def transform(
        self, function: Union[Callable, List[Callable]], targets: bool = False
    ) -> Union["HatecompDataset", "HatecompDatasetView"]:
        if not isinstance(function, (tuple, list)):
            function = [function]
        if targets:
            self.target_transforms += function
            self.has_target_transforms = True
        else:
            self.data_transforms += function
            self.has_data_transforms = True
        return self

    def __add__(self, object):
        if not isinstance(object, FunctionalDataset):
            raise AttributeError(f"Cannot add a dataset to {type(object)}")

        return ConcatHatecompDatasetView(self, object)

    def num_classes():
        raise NotImplementedError

    def examples(self, num_examples: int = 5):
        return [self[i] for i in range(num_examples)]


class HatecompDataset(FunctionalDataset):
    def __init__(self, root: str = None, download: bool = True) -> None:
        super().__init__()
        self.DEFAULT_DIRECTORY = os.path.join(
            hatecomp.__location__, "datasets/data", type(self).__name__, "data"
        )
        if root is None:
            logging.info(
                f"{type(self).__name__} root is not set, using the default data root of {self.DEFAULT_DIRECTORY}"
            )
            root = self.DEFAULT_DIRECTORY

        try:
            self.data = self.prepare_data(root)
        except FileNotFoundError as e:
            logging.warning(f"Data could not be loaded from {root}.")
            if download:
                logging.info(
                    f"Downloading {type(self).__name__} data to location {root}."
                )
                if not os.path.exists(root):
                    os.makedirs(root)
                self.download(root)
                self.data = self.prepare_data(root)
            else:
                raise e
        logging.info(f"Loaded {type(self).__name__} from {root}.")

        self.setup()
        logging.info(f"Setup {type(self).__name__}.")

    def prepare_data(self, path: str) -> List["HatecompDataItem"]:
        raise NotImplementedError

    def download(self, path: str):
        raise NotImplementedError

    def setup(self):
        pass

    def map(
        self,
        function: Union[Callable, List[Callable]],
        targets: bool = False,
        batch_size: int = None,
    ) -> Union["HatecompDataset", "HatecompDatasetView"]:
        if targets:
            attribute = "target"
        else:
            attribute = "data"

        if batch_size is None:
            for idx, example in enumerate(self.data):
                data_item = self.data[idx]
                setattr(data_item, attribute, function(getattr(data_item, attribute)))
        else:
            for slice, batch in batch_enumerate(self.data, batch_size):
                mapped_batch_data = function(
                    [getattr(data_item, attribute) for data_item in batch]
                )
                for idx, mapped_example_data in zip(slice, mapped_batch_data):
                    data_item = self.data[idx]
                    setattr(data_item, attribute, mapped_example_data)

        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            data_item = self.data[index]
            id, data, target = data_item.id, data_item.data, data_item.target
            if id is None:
                id = f"{type(self).__name__}-{index}"
            if self.has_data_transforms:
                data = map_functions(data, self.data_transforms)
            if self.has_target_transforms:
                target = map_functions(target, self.target_transforms)
            return {
                "id": id,
                "data": data,
                "target": target,
            }
        elif isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return HatecompDatasetView(self, data_indices, sorted=False)
        elif isinstance(index, (tuple, list)):
            return HatecompDatasetView(self, index)


class HatecompDatasetView(FunctionalDataset):
    def __init__(
        self,
        dataset: Union[HatecompDataset, "HatecompDatasetView"],
        view_indices: List[int],
        sorted=True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        unique_indices = get_unique(view_indices, ordered=sorted)
        self.data_indices = unique_indices

    def map(self, function: Callable, targets: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over a dataset view!")

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        if isinstance(index, int):
            data_item = self.dataset[self.data_indices[index]]
            id, data, target = data_item["id"], data_item["data"], data_item["target"]
            if self.has_data_transforms:
                data = map_functions(data, self.data_transforms)
            if self.has_target_transforms:
                target = map_functions(target, self.target_transforms)
            return {
                "id": id,
                "data": data,
                "target": target,
            }
        elif isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return HatecompDatasetView(self, data_indices, sorted=False)
        elif isinstance(index, (tuple, list)):
            return HatecompDatasetView(self, index)


class ConcatHatecompDatasetView(FunctionalDataset):
    def __init__(
        self,
        datatset_one: Union[HatecompDataset, "HatecompDatasetView"],
        dataset_two: Union[HatecompDataset, "HatecompDatasetView"],
    ):
        super().__init__()
        self.dataset_one = datatset_one
        self.dataset_two = dataset_two
        self.transition_point = len(datatset_one)

    def map(self, function: Callable, targets: bool = False, batch_size: int = None):
        raise AttributeError("Cannot map over concatenated datasets!")

    def __len__(self):
        return len(self.dataset_one) + len(self.dataset_two)

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < self.transition_point:
                selected_dataset = self.dataset_one
            else:
                selected_dataset = self.dataset_two
                index -= self.transition_point
            data_item = selected_dataset[index]
            id, data, target = data_item["id"], data_item["data"], data_item["target"]
            if self.has_data_transforms:
                data = map_functions(data, self.data_transforms)
            if self.has_target_transforms:
                target = map_functions(target, self.target_transforms)
            return {
                "id": id,
                "data": data,
                "target": target,
            }
        elif isinstance(index, slice):
            data_indices = list(range(len(self))[index])
            return HatecompDatasetView(self, data_indices, sorted=False)
        elif isinstance(index, (tuple, list)):
            return HatecompDatasetView(self, index)


class HatecompDataItem:
    __slots__ = ("id", "data", "target")

    def __init__(self, data, id=None, target=None) -> None:
        self.data = data
        self.id = id
        self.target = target

    def __iter__(self):
        return iter((self.id, self.data, self.target))
