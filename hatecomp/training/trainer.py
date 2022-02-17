from typing import Optional
from dataclasses import dataclass, field

import torch
from transformers.trainer import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from hatecomp.base.utils import get_class_weights
from transformers.optimization import (
    AdamW,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


@dataclass
class HatecompTrainingArgs(TrainingArguments):
    lr_cycles: Optional[int] = field(
        default=1,
        metadata={
            "help": "Sets the number of cycles for the learning rate scheduler to complete over training"
        },
    )


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

    # def create_optimizer(self):
    #     optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
    #         self.args
    #     )
    #     self.optimizer = optimizer_cls(
    #         self.model.parameters(),
    #         **optimizer_kwargs,
    #     )
    #     return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                num_cycles=self.args.lr_cycles,
            )
