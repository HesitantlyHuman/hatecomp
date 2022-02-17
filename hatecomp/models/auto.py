# Use the implementation of https://github.com/shahrukhx01/multitask-learning-transformers/tree/main/multiple_prediction_heads to allow for multitask
# Figure out how to update the datasets which have multiple tasks for this

from transformers import AutoModelForSequenceClassification


class HatecompAutoModelForSequenceClassification(AutoModelForSequenceClassification):
    def from_pretrained(transformer_name: str, num_labels: int):
        model = AutoModelForSequenceClassification.from_pretrained(
            transformer_name, num_labels=num_labels
        )
        if len(num_labels) > 1:
            model.config.multiheaded = True
        else:
            model.config.multiheaded = False

        print(model.classification_head)
