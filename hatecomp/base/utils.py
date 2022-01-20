from typing import Callable
import torch

def tokenize_bookends(string_in: str, output_length: int, tokenizer: Callable):
    end_token_length = int(output_length / 2)
    start_token_length = output_length - end_token_length
    tokens = tokenizer(string_in)
    return {
        mask : get_bookends(torch.tensor(tensor), start_token_length, end_token_length)
        for mask, tensor in tokens.items()
    }
    
def get_bookends(sequence, n_start, n_end):
    return torch.cat((sequence[:n_start], sequence[-n_end:]), dim = 0)