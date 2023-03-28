from typing import List, Tuple

import os
import csv

import torch

from hatecomp.datasets.base.data import _HatecompDataset
from hatecomp.datasets.MLMA.download import MLMADownloader
from hatecomp._path import install_path


class MLMADataset(_HatecompDataset):
    __name__ = "MLMA"
    DEFAULT_DIRECTORY = os.path.join(install_path, "datasets/MLMA/data")
    DOWNLOADER = MLMADownloader

    HOSTILITY_SENTIMENTS = [
        "abusive",
        "hateful",
        "offensive",
        "disrespectful",
        "fearful",
        "normal",
    ]
    ANNOTATOR_SENTIMENTS = [
        "disgust",
        "shock",
        "anger",
        "sadness",
        "fear",
        "confusion",
        "indifference",
    ]
    LABEL_KEY = {
        "indirect": 0,
        "direct": 1,
        "origin": 0,
        "gender": 1,
        "sexual_orientation": 2,
        "religion": 3,
        "disability": 4,
        "other": 5,
        "individual": 0,
        "women": 1,
        "special_needs": 2,
        "african_descent": 3,
        "immigrants": 4,
        "refugees": 4,
        "muslims": 5,
        "arabs": 5,
        "indian/hindu": 6,
        "jews": 7,
        "hispanics": 8,
        "asians": 9,
        "left_wing_people": 10,
        "gay": 11,
        "christian": 12,
    }

    def load_data(self, path: str) -> Tuple[list, list, list]:
        path = os.path.join(path, "hate_speech_mlma/en_dataset_with_stop_words.csv")
        ids = []
        data = []
        labels = []
        with open(path) as file:
            for row in list(csv.reader(file))[1:]:
                ids.append(row[0])
                data.append(row[1])
                labels.append(row[2:])
        return (ids, data, labels)

    def encode_labels(self, encoding_scheme: dict) -> List:
        encoded_multiclass = []
        for labels in self.labels:
            encoded_labels = []
            sentiment, directness, annotator_sentiment, target, group = labels
            sentiments = sentiment.split("_")
            for sentiment in sentiments:
                assert sentiment in self.HOSTILITY_SENTIMENTS, sentiment
            encoded_labels += [
                int(sentiment in sentiments) for sentiment in self.HOSTILITY_SENTIMENTS
            ]
            encoded_labels.append(self.LABEL_KEY[directness])
            sentiments = annotator_sentiment.split("_")
            for sentiment in sentiments:
                assert sentiment in self.ANNOTATOR_SENTIMENTS, sentiment
            encoded_labels += [
                int(sentiment in sentiments) for sentiment in self.ANNOTATOR_SENTIMENTS
            ]
            encoded_labels.append(self.LABEL_KEY[target])
            encoded_labels.append(self.LABEL_KEY[group])
            encoded_multiclass.append(torch.squeeze(torch.tensor(encoded_labels)))
        self.num_classes = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 13]
        return torch.stack(encoded_multiclass, dim=0)
