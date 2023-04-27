from typing import Callable
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate


def id_collate(unprocessed_batch):
    batch = {}
    for key, value in unprocessed_batch[0].items():
        if key == "label":
            batch_key = "labels"
        else:
            batch_key = key
        if isinstance(value, torch.Tensor):
            batch[batch_key] = torch.stack(
                [batch_item[key] for batch_item in unprocessed_batch]
            )
        elif isinstance(value, str):
            batch[batch_key] = [batch_item[key] for batch_item in unprocessed_batch]
        elif isinstance(value, np.ndarray):
            batch[batch_key] = np.array(
                [batch_item[key] for batch_item in unprocessed_batch]
            )
            batch[batch_key] = torch.from_numpy(batch[batch_key])
        else:
            batch[batch_key] = torch.tensor(
                [batch_item[key] for batch_item in unprocessed_batch]
            )
    return batch


def batch_and_slice(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        upper = min(ndx + batch_size, length)
        yield (slice(ndx, upper), iterable[ndx:upper])


def get_class_weights(dataset, n_classes):
    binary_target_idxs = []
    label_counts = []
    for idx, n_classes_task in enumerate(n_classes):
        if n_classes_task == 1:
            label_counts.append(torch.zeros(2))
            binary_target_idxs.append(idx)
        else:
            label_counts.append(torch.zeros(n_classes_task))

    for item in dataset:
        labels = item["label"]

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        if len(labels.shape) == 0:
            labels = labels.unsqueeze(0)

        for idx, label in enumerate(labels):
            if idx in binary_target_idxs:
                label_counts[idx][0] += 1 - label
                label_counts[idx][1] += label
            else:
                label = label.long()
                label_counts[idx][label] += 1

    weights = []
    for idx, label_count in enumerate(label_counts):
        if idx in binary_target_idxs:
            weights.append(label_count[0] / label_count[1])
        else:
            inverted_counts = 1 / label_count
            weights.append(inverted_counts / torch.mean(inverted_counts))

    return weights
