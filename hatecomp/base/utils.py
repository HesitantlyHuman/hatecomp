from typing import Callable
import torch
from torch.utils.data.dataloader import default_collate


def tokenize_bookends(string_in: str, output_length: int, tokenizer: Callable):
    end_token_length = int(output_length / 2)
    start_token_length = output_length - end_token_length
    tokens = tokenizer(string_in)
    return {
        mask: get_bookends(torch.tensor(tensor), start_token_length, end_token_length)
        for mask, tensor in tokens.items()
    }


def get_bookends(sequence, n_start, n_end):
    return torch.cat((sequence[:n_start], sequence[-n_end:]), dim=0)


def id_collate(unprocessed_batch):
    batch_without_ids = []
    ids = []
    for example in unprocessed_batch:
        batch_without_ids.append(
            {key: value for key, value in example.items() if not key == "id"}
        )
        ids.append(example["id"])
    processed_batch = default_collate(batch_without_ids)
    processed_batch.update({"id": ids})
    return processed_batch
