from typing import Optional, List, Dict, Union, Any, Tuple
from dataclasses import dataclass, field

import torch
from transformers.trainer import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from hatecomp.datasets.base.utils import get_class_weights, id_collate
from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers.utils import logging
from transformers.trainer_pt_utils import nested_detach


logger = logging.get_logger(__name__)


@dataclass
class HatecompTrainingArgs(TrainingArguments):
    lr_cycles: Optional[int] = field(
        default=1,
        metadata={
            "help": "Sets the number of cycles for the learning rate scheduler to complete over training"
        },
    )


class HatecompTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args: TrainingArguments = None,
        data_collator=None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
    ):
        if data_collator is None:
            data_collator = id_collate
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        self.class_weights = [
            task_weights.to(self.args.device)
            for task_weights in get_class_weights(
                self.train_dataset, self.model.config.num_labels
            )
        ]
        return super().get_train_dataloader()

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                loss = loss.mean().detach()

                logits = []
                for task_outputs in outputs:
                    if isinstance(task_outputs, dict):
                        logits.append(
                            tuple(
                                v
                                for k, v in task_outputs.items()
                                if k not in ignore_keys + ["loss"]
                            )
                        )
                    else:
                        logits.append(task_outputs[1:])
            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)

                logits = []
                for task_outputs in outputs:
                    if isinstance(task_outputs, dict):
                        logits.append(
                            tuple(
                                v
                                for k, v in task_outputs.items()
                                if k not in ignore_keys
                            )
                        )
                    else:
                        logits.append(task_outputs)
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = [
                        task_outputs[self.args.past_index - 1]
                        for task_outputs in outputs
                    ]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        outputs = model(**{k: v for k, v in inputs.items() if not k == "id"})

        losses = []
        for task_idx, task_outputs in enumerate(outputs):
            logits = task_outputs.get("logits")

            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights[task_idx])
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels[task_idx]),
                labels[:, task_idx].view(-1),
            )
            losses.append(loss)
        return (
            (torch.sum(torch.stack(losses)), outputs)
            if return_outputs
            else torch.sum(torch.stack(losses))
        )

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
