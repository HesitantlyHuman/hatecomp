from typing import Callable
import torch
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
    label_counts = [torch.zeros(n_classes_task) for n_classes_task in n_classes]
    for batch in dataset:
        labels = batch["label"]

        if isinstance(labels, torch.Tensor):
            labels = labels.long()
        else:
            labels = torch.tensor(labels).long()

        if len(labels.shape) == 0:
            label_counts[0][labels] += 1
        else:
            for idx, label in enumerate(labels):
                label_counts[idx][label] += 1

    inverted_classes = [1 / label_count for label_count in label_counts]
    return [
        inverted_class / torch.mean(inverted_class)
        for inverted_class in inverted_classes
    ]
