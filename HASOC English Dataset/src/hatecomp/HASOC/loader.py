import os
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

DOWNLOAD_DIR = './HASOC English Dataset/data/'
DATA_RELATIVE_PATH = '/english_dataset'

TSVs = [
    'english_dataset.tsv',
    'hasoc2019_en_test-2919.tsv'
]

class HASOCDataset(Dataset):
    def __init__(
        self,
        data_dir = None
    ):
        if data_dir is None:
            data_dir = os.path.join(DOWNLOAD_DIR, DATA_RELATIVE_PATH)

        self.data_dir = data_dir
        self.data = self._load(self.data_dir)

    def _load(path: str) -> List[Tuple]:
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return

