# Use the implementation of https://github.com/shahrukhx01/multitask-learning-transformers/tree/main/multiple_prediction_heads to allow for multitask
# Figure out how to update the datasets which have multiple tasks for this
import torch
from transformers import AutoModelForSequenceClassification


class HatecompAutoModelForSequenceClassification(AutoModelForSequenceClassification):
    def from_pretrained(transformer_name: str, num_labels: int):
        model = AutoModelForSequenceClassification.from_pretrained(
            transformer_name, num_labels=1
        )
        if isinstance(num_labels, int):
            model.config.multiheaded = False
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
        else:
            model.config.multiheaded = False

        if model.config.multiheaded:
            model = HatecompAutoModelForSequenceClassification.recapitate(
                model, num_labels=num_labels
            )

        return model

    # TODO
    def recapitate(model, num_labels):
        return model
