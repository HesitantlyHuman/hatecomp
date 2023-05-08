from typing import Callable, List
import os
import json
import logging

import torch
from transformers import AutoTokenizer

from hatecomp.models.download import verify_pretrained_download, PRETRAINED_INSTALLATION_LOCATION

# Suppress the too long message for tokinization, since we will handle that ourselves
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class HatecompTokenizer:
    def __init__(
        self,
        tokenizer: Callable[[List[str]], List[int]],
        max_length: int = 512,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self._start_token_length = int(max_length / 2)
        self._end_token_length = max_length - self._start_token_length

    def __call__(self, tokenization_input: str):
        tokens = self.tokenizer(
            tokenization_input, padding=True, truncation=False, return_tensors="pt"
        )

        # If the length of the tokens is greater than the max_length, then
        # grab the first half of the tokens and the last half of the tokens
        # and concatenate them together.
        if tokens["input_ids"].shape[1] > self.max_length:
            for key in tokens:
                tokens[key] = torch.cat(
                    (
                        tokens[key][:, : self._start_token_length],
                        tokens[key][:, -self._end_token_length :],
                    ),
                    dim=1,
                )

        return tokens

    @classmethod
    def from_huggingface_pretrained(
        self, model_name: str, max_length: int = 512
    ) -> "HatecompTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return HatecompTokenizer(tokenizer, max_length=max_length)
    
    @classmethod
    def from_hatecomp_pretrained(
        self,
        pretrained_model_name_or_path: str,
        download: bool = False,
        force_download: bool = False
    ) -> "HatecompTokenizer":
        pretrained_model_name_or_path = pretrained_model_name_or_path.lower()

        verify_pretrained_download(
            pretrained_model_name_or_path, download, force_download
        )

        # Load the model configuration
        config_path = os.path.join(
            PRETRAINED_INSTALLATION_LOCATION,
            pretrained_model_name_or_path,
            "config.json",
        )
        with open(config_path, "r") as f:
            config = json.load(f)

        return self.from_huggingface_pretrained(
            config["transformer_name"], max_length=config["max_length"]
        )



if __name__ == "__main__":
    tokenizer = HatecompTokenizer.from_hatecomp_pretrained("MLMA", force_download=True)