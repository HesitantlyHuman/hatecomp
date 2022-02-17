# Use the implementation of https://github.com/shahrukhx01/multitask-learning-transformers/tree/main/multiple_prediction_heads to allow for multitask
# Figure out how to update the datasets which have multiple tasks for this
import torch
from transformers import AutoModelForSequenceClassification


class HatecompAutoModelForSequenceClassification(AutoModelForSequenceClassification):
    def from_pretrained(transformer_name: str, num_labels: int):
        model = AutoModelForSequenceClassification.from_pretrained(
            transformer_name, num_labels=num_labels
        )
        if isinstance(num_labels, int):
            model.config.multiheaded = False
        elif isinstance(num_labels, (tuple, list)):
            if len(num_labels) > 1:
                model.config.multiheaded = True
            else:
                model.config.multiheaded = False
        elif isinstance(num_labels, torch.Tensor):
            

        print(model.classification_head)
