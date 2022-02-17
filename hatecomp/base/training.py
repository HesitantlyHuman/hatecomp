import torch
from transformers.trainer import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from hatecomp.base.utils import get_class_weights


class HatecompTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        self.class_weights = get_class_weights(
            self.train_dataset, self.model.config.num_labels
        ).to(self.args.device)
        return super().get_train_dataloader()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
