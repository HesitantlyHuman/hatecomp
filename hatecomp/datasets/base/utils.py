from typing import Callable, Iterable
import torch
from torch.utils.data.dataloader import default_collate


def tokenize_bookends(tokenization_input: str, output_length: int, tokenizer: Callable):
    end_token_length = int(output_length / 2)
    start_token_length = output_length - end_token_length
    tokens = tokenizer(tokenization_input, padding="max_length", truncation=False)
    example_key = list(tokens.keys())[0]
    outputs = [{} for i in range(len(tokens[example_key]))]
    for key, batch in tokens.items():
        for idx, data in enumerate(batch):
            outputs[idx].update(
                {
                    key: get_bookends(
                        torch.tensor(data), start_token_length, end_token_length
                    )
                }
            )
    return outputs


def get_bookends(sequence, length_start, length_end):
    return torch.cat((sequence[:length_start], sequence[-length_end:]), dim=0)


def batch(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        upper = min(ndx + batch_size, length)
        yield iterable[ndx:upper]


def batch_enumerate(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        upper = min(ndx + batch_size, length)
        yield (slice(ndx, upper), iterable[ndx:upper])


def get_unique(input_iterator, ordered=True):
    if ordered:
        unique = list(set(input_iterator))
        unique.sort()
        return unique
    else:
        seen = set()
        seen_add = seen.add
        return [item for item in input_iterator if not (item in seen or seen_add(item))]


def map_functions(obj: object, function_list: Iterable[Callable]):
    value = obj
    for function in function_list:
        value = function(value)
    return value


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
