from typing import List
import os
import json
import logging

import torch
from transformers import AutoModel

from hatecomp.models.download import verify_pretrained_download, PRETRAINED_INSTALLATION_LOCATION

# Suppress the huggingface warning about fine tuning the model
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class HatecompClassifier(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        heads: List[torch.nn.Module],
    ):
        super().__init__()
        self.transformer = base_model
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, *args, **kwargs):
        kwargs = {
            parameter: value
            for parameter, value in kwargs.items()
            if not parameter == "labels"
        }
        # Get the CLS token output
        transformer_output = self.transformer(*args, **kwargs)[0][:, 0, :]
        return [head(transformer_output) for head in self.heads]

    @classmethod
    def from_huggingface_pretrained(
        self,
        transformer_name: str,
        num_classes: List[int],
        head_hidden_size: int = 768,
        dropout: float = 0.1,
    ) -> "HatecompClassifier":
        transformer = AutoModel.from_pretrained(transformer_name)
        heads = [
            HatecompClassificationHead(
                transformer.config.hidden_size,
                head_hidden_size,
                num_classes=task_num_classes,
                dropout=dropout,
            )
            for task_num_classes in num_classes
        ]
        return self(transformer, heads)

    @classmethod
    def from_hatecomp_pretrained(
        self,
        pretrained_model_name_or_path: str,
        download: bool = False,
        force_download: bool = False,
    ) -> "HatecompClassifier":
        pretrained_model_name_or_path = pretrained_model_name_or_path.lower()

        # Download the model if it doesn't exist
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

        # Create a new model
        model = self.from_huggingface_pretrained(
            config["transformer_name"],
            config["num_classes"],
            config["head_hidden_size"],
            config["dropout"],
        )

        # Load the state dict
        state_dict_path = os.path.join(
            PRETRAINED_INSTALLATION_LOCATION, pretrained_model_name_or_path, "model.pt"
        )
        model.load_state_dict(torch.load(state_dict_path))

        # Return the model
        return model


class HatecompClassificationHead(torch.nn.Module):
    def __init__(
        self,
        transformer_hidden_size: int,
        head_hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dense = torch.nn.Linear(transformer_hidden_size, head_hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.out_proj = torch.nn.Linear(head_hidden_size, num_classes)

    def forward(self, features, **kwargs):
        x = features
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


if __name__ == "__main__":
    model = HatecompClassifier.from_hatecomp_pretrained("MLMA", force_download=True)
    print(model)
