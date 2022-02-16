from typing import List, Tuple

import os
import csv
import logging

import numpy as np

from hatecomp.base.data import _HateDataset
from hatecomp.datasets.MLMA.download import MLMADownloader


class MLMADataset(_HateDataset):
    __name__ = "MLMA"
    downloader = MLMADownloader

    def _load_data(self, path: str) -> Tuple[List]:
        path = os.path.join(path, "hate_speech_mlma/en_dataset_with_stop_words.csv")
        ids = []
        data = []
        labels = []
        with open(path) as file:
            for row in list(csv.reader(file))[1:]:
                ids.append(row[0])
                data.append(row[1])
                labels.append(row[2:])
        return (np.array(ls) for ls in [ids, data, labels])
