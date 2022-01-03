from typing import List, Tuple

import os
import csv

import numpy as np

from hatecomp.base.dataset import _HateDataset
from hatecomp.datasets.ZeerakW.download import ZeerakWDownloader

class NAACLDataset(_HateDataset):
    __name__ = 'NAACL'
    downloader = ZeerakWDownloader

    PATH = 'hatecomp/datasets/ZeerakW/data/NAACL_SRW_2016.csv'

    CSV_FILES = [
        'NAACL_SRW_2016.csv'
    ]
    ENCODING_KEY = {
        'none' : 0,
        'racism' : 1,
        'sexism' : 2
    }

    def __init__(
        self,
        path = None,
        one_hot = True
    ):
        self.one_hot = one_hot
        super(NAACLDataset, self).__init__(path = self.PATH)
        
    def _load_data(self) -> Tuple[List]:
        returns = ([], [], [])
        for tsv_file in NAACLDataset.CSV_FILES:
            tsv_path = os.path.join(path, tsv_file)
            tsv = self._read_tsv(tsv_path)
            returns = returns + self._convert_tsv(tsv)
        return returns

    def _read_tsv(self, path: str) -> List[List[str]]:
        with open(path) as file:
            return list(csv.reader(file))

    def _convert_tsv(self, tsv: List[List[str]]) -> Tuple[List]:
        ids = []
        data = []
        labels = []
        for row in tsv[1:]:
            ids.append(row[0])
            data.append(row[1])
            labels.append([NAACLDataset.ENCODING_KEY[encoding] for encoding in row[2:]])
        return tuple(np.array(data_list) for data_list in [ids, data, labels])
        

if __name__ == '__main__':
    dataset = NAACLDataset()