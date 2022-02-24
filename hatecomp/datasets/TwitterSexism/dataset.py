from typing import List, Tuple

import os
import csv

from hatecomp.datasets.base import HatecompDataset, HatecompDataItem
from hatecomp.datasets.TwitterSexism.download import TwitterSexismDownloader


class TwitterSexismDataset(HatecompDataset):
    CSV_FILE = "twitter_sexism.csv"
    LABEL_KEY = {"benevolent": 0, "hostile": 1}

    def __init__(self, root: str = None, download: bool = True):
        super().__init__(root=root, download=download)

    def prepare_data(self, path: str) -> Tuple[HatecompDataItem]:
        with open(os.path.join(path, self.CSV_FILE)) as data_file:
            csv_data = list(csv.reader(data_file))
        data_items = []
        for item in csv_data[1:]:
            data_items.append(
                HatecompDataItem(item[1], id=item[0], target=self.LABEL_KEY[item[2]])
            )
        return data_items

    def download(self, path: str) -> None:
        downloader = TwitterSexismDownloader(path)
        downloader.load()
