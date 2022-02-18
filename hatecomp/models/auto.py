from typing import List
import torch
from transformers import AutoModelForSequenceClassification
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class HatecompConfig:
    OVERRIDDEN_ATTRIBUTES = ["num_labels"]

    def __init__(self, config: PretrainedConfig):
        self.config = config

    def ___setattr__(self, key, value):
        if key in self.OVERRIDDEN_ATTRIBUTES:
            super().__setattr__(key, value)
        else:
            self.config.__setattr__(key, value)

    def __getattr__(self, key):
        if key in self.OVERRIDDEN_ATTRIBUTES:
            return getattr(self, key)
        else:
            return getattr(self.config, key)


class HatecompMultiheaded(torch.nn.Module):
    def __init__(
        self, base_model: torch.nn.Module, heads: List[torch.nn.Module], config
    ):
        super().__init__()
        self.base = base_model
        self.heads = torch.nn.ModuleList(heads)
        self.config = config

    def forward(self, *args, **kwargs):
        kwargs = {
            parameter: value
            for parameter, value in kwargs.items()
            if not parameter == "labels"
        }
        base_outputs = self.base(*args, **kwargs)
        return [
            SequenceClassifierOutput(
                loss=None,
                logits=head(base_outputs),
                hidden_states=base_outputs.hidden_states,
                attentions=base_outputs.attentions,
            )
            for head in self.heads
        ]


# Copied from the huggingface RobertaClassificationHead
class HatecompClassificationHead(torch.nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.out_proj = torch.nn.Linear(config.hidden_size, num_classes)

    def forward(self, features, **kwargs):
        x = features[0][:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HatecompAutoModelForSequenceClassification(AutoModelForSequenceClassification):
    def from_pretrained(transformer_name: str, num_labels: List[int]):
        model = AutoModelForSequenceClassification.from_pretrained(
            transformer_name, num_labels=1
        )
        if isinstance(num_labels, int):
            model.config.multiheaded = False
            num_labels = [num_labels]
        elif isinstance(num_labels, (tuple, list)):
            if len(num_labels) > 1:
                model.config.multiheaded = True
            else:
                model.config.multiheaded = False
        elif isinstance(num_labels, torch.Tensor):
            if len(num_labels.shape) == 0:
                model.config.multiheaded = False
            else:
                model.config.multiheaded = True

        model = HatecompAutoModelForSequenceClassification.recapitate(
            model, num_labels=num_labels
        )

        return model

    def recapitate(model, num_labels):
        config = HatecompConfig(model.config)
        config.num_labels = num_labels
        classifiers = torch.nn.ModuleList(
            [
                HatecompClassificationHead(config, num_classes)
                for num_classes in num_labels
            ]
        )
        base_model = getattr(model, model.base_model_prefix)
        return HatecompMultiheaded(base_model, classifiers, config)
