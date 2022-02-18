# Use the implementation of https://github.com/shahrukhx01/multitask-learning-transformers/tree/main/multiple_prediction_heads to allow for multitask
# Figure out how to update the datasets which have multiple tasks for this
import torch
from transformers import AutoModelForSequenceClassification

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
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HatecompAutoModelForSequenceClassification(AutoModelForSequenceClassification):
    def from_pretrained(transformer_name: str, num_labels: int):
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
        model.classifier = torch.nn.ModuleList(
            [
                HatecompClassificationHead(model.config, num_classes)
                for num_classes in num_labels
            ]
        )
        return model
