from typing import Callable
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


def id_collate(unprocessed_batch):
    batch_without_ids = []
    ids = []
    for id, data, label in unprocessed_batch:
        batch_without_ids.append((data, label))
        ids.append(id)
    processed_data, processed_labels = default_collate(batch_without_ids)
    return (ids, processed_data, processed_labels[0])


def batch_and_slice(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        upper = min(ndx + batch_size, length)
        yield (slice(ndx, upper), iterable[ndx:upper])
