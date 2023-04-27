from typing import List, Tuple

import re
import os
import csv

import numpy as np

from hatecomp._path import install_path
from hatecomp.datasets.base.data import _HatecompDataset
from hatecomp.datasets.WikiTalk.download import (
    WikiToxicityDownloader,
    WikiAggressionDownloader,
    WikiPersonalAttacksDownloader,
)


class _WikiDataset(_HatecompDataset):
    DEFAULT_DIRECTORY = os.path.join(install_path, "datasets/WikiTalk/data")
    LABEL_KEY = ""

    def __init__(self, root: str = None, download: bool = True):
        super().__init__(root=root, download=download)

    def normalize_label(self, label: float) -> float:
        raise NotImplementedError

    def load_data(self, path: str) -> Tuple[List]:
        # Load annotations
        annotations = {}
        with open(os.path.join(path, self.TSVs[0])) as data_file:
            csv_data = list(csv.reader(data_file, delimiter="\t"))
        for item in csv_data[1:]:
            item_id = int(float(item[0]))
            if item_id not in annotations:
                annotations[item_id] = []
            annotations[item_id].append(float(item[3]))

        # Compute labels
        for item_id, annotation in annotations.items():
            annotations[item_id] = self.normalize_label(np.mean(annotation))

        # Load comments
        comments = {}
        with open(os.path.join(path, self.TSVs[1])) as data_file:
            csv_data = list(csv.reader(data_file, delimiter="\t"))
        for item in csv_data[1:]:
            item_id = int(float(item[0]))
            comments[item_id] = clean_wiki_comment(item[1])

        # Combine comments and labels
        ids, data, labels = [], [], []
        for item_id, comment in comments.items():
            ids.append(item_id)
            data.append(comment)
            labels.append(annotations[item_id])

        return (np.array(ids), data, np.array(labels))

    def encode_labels(self, encoding_scheme: dict) -> List:
        self.num_classes = [1]
        return self.labels


class WikiToxicityDataset(_WikiDataset):
    __name__ = "WikiToxicity"
    DOWNLOADER = WikiToxicityDownloader
    TSVs = ["toxicity_annotations.tsv", "toxicity_annotated_comments.tsv"]

    def normalize_label(self, label: float) -> float:
        return (label + 2) / 4


class WikiAggressionDataset(_WikiDataset):
    __name__ = "WikiAggression"
    DOWNLOADER = WikiAggressionDownloader
    TSVs = ["aggression_annotations.tsv", "aggression_annotated_comments.tsv"]

    def normalize_label(self, label: float) -> float:
        return (label + 3) / 6


class WikiPersonalAttacksDataset(_WikiDataset):
    __name__ = "WikiPersonalAttacks"
    DOWNLOADER = WikiPersonalAttacksDownloader
    TSVs = ["attack_annotations.tsv", "attack_annotated_comments.tsv"]

    def load_data(self, path: str) -> Tuple[List]:
        # Load annotations
        annotations = {}
        with open(os.path.join(path, self.TSVs[0])) as data_file:
            csv_data = list(csv.reader(data_file, delimiter="\t"))
        for item in csv_data[1:]:
            item_id = int(float(item[0]))
            if item_id not in annotations:
                annotations[item_id] = [[], [], [], []]
            annotations[item_id][0].append(float(item[2]))
            annotations[item_id][1].append(float(item[3]))
            annotations[item_id][2].append(float(item[4]))
            annotations[item_id][3].append(float(item[5]))

        # Compute labels
        for item_id, annotation in annotations.items():
            annotations[item_id] = [
                np.mean(annotation[0]),
                np.mean(annotation[1]),
                np.mean(annotation[2]),
                np.mean(annotation[3]),
            ]

        # Load comments
        comments = {}
        with open(os.path.join(path, self.TSVs[1])) as data_file:
            csv_data = list(csv.reader(data_file, delimiter="\t"))
        for item in csv_data[1:]:
            item_id = int(float(item[0]))
            comments[item_id] = clean_wiki_comment(item[1])

        # Combine comments and labels
        ids, data, labels = [], [], []
        for item_id, comment in comments.items():
            ids.append(item_id)
            data.append(comment)
            labels.append(annotations[item_id])

        return (np.array(ids), data, np.array(labels))

    def encode_labels(self, encoding_scheme: dict) -> List:
        self.num_classes = [1, 1, 1, 1]
        return self.labels


def clean_wiki_comment(comment: str) -> str:
    comment = str(comment)

    # Replace NEWLINE_TOKEN with newline
    comment = comment.replace("NEWLINE_TOKEN", "\n")
    # Replace TAB_TOKEN with tab
    comment = comment.replace("TAB_TOKEN", "\t")

    comment = comment.replace("\r", " ")
    regex = re.compile(r"\s*\n\s*")
    comment = regex.sub("\n", comment)
    regex = re.compile(r"[\t \r]*\t[\t \r]*")
    comment = regex.sub("\t", comment)
    regex = re.compile(r"[ \r]* [ \r]*")
    comment = regex.sub(" ", comment)
    comment = comment.strip()

    return comment


if __name__ == "__main__":
    dataset = WikiPersonalAttacksDataset()
    print(dataset[0])
    print(dataset.num_classes)
