from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
import uuid
import os

import torch

from hatecomp.models import HatecompClassifier, HatecompTokenizer
from hatecomp.training import HatecompTrainer
from hatecomp.datasets.base.data import _HatecompDataset
from hatecomp.datasets.base import DataLoader
from hatecomp.datasets.base.utils import get_class_weights

PARAMETERS_FILENAME = "trial_parameters.json"


# Holds parameters for a single trial
@dataclass
class HatecompTrialConfig:
    name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the trial. If not specified, a random name will be generated. If specified, the trial will be loaded from the checkpoint directory if it exists."
        },
    )
    transformer_name: Optional[str] = field(
        default="roberta-base",
        metadata={
            "help": "Name of the transformer model to use as the backbone of the multiheaded classifier"
        },
    )
    epochs: Optional[int] = field(
        default=3,
        metadata={"help": "Number of epochs to train the model for"},
    )
    training_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size for training"},
    )
    evaluation_batch_size: Optional[int] = field(
        default=64,
        metadata={"help": "Batch size for evaluation"},
    )
    learning_rate: Optional[float] = field(
        default=1e-4,
        metadata={
            "help": "Maximum rate for the AdamW optimizer during the learning rate schedule"
        },
    )
    weight_decay: Optional[float] = field(
        default=0.1,
        metadata={"help": "Weight decay for the AdamW optimizer"},
    )
    learning_rate_warmup_percentage: Optional[float] = field(
        default=0.3,
        metadata={
            "help": "Percentage of the training during which the OneCycleLR learning rate schedule will increase to the maximum learning rate"
        },
    )
    dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the multiheaded classifier"},
    )
    head_hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "Hidden size for the multiheaded classifier"},
    )
    device: Optional[str] = field(
        default="cuda",
        metadata={"help": "Device to use for training"},
    )


class HatecompTrialRunner:
    def __init__(
        self,
        root: str,
        dataset: _HatecompDataset,
        configuration: HatecompTrialConfig,
        verbose: bool = True,
        checkpoint: bool = True,
    ):
        self.root = root
        self.verbose = verbose
        self.dataset = dataset
        self.configuration = configuration
        self.checkpoint = checkpoint

        if self.configuration.name is None:
            # Generate a random 5-character name for the trial
            self.configuration.name = str(uuid.uuid4())[:5]

        if self.verbose:
            print(f"Starting new trial with name: {self.configuration.name}")

        self.trial_dir = os.path.join(self.root, self.configuration.name)

        self.load_or_save_parameters()

        (
            self.training_dataloader,
            self.test_dataloader,
            self.class_weights,
        ) = self.get_dataloaders(self.dataset)
        self.tokenizer, self.model = self.get_tokenizer_and_model()
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler(self.model)
        self.loss_function = self.get_loss_function()

    def load_or_save_parameters(self) -> None:
        # Load the parameters if they exist, save them otherwise
        parameters_path = os.path.join(self.trial_dir, PARAMETERS_FILENAME)
        if os.path.exists(parameters_path):
            with open(parameters_path, "r") as parameters_file:
                parameters = json.load(parameters_file)
            for key, value in parameters.items():
                setattr(self.configuration, key, value)
        else:
            os.makedirs(self.trial_dir, exist_ok=True)
            with open(parameters_path, "w") as parameters_file:
                json.dump(self.configuration.__dict__, parameters_file, indent=4)

    def get_dataloaders(
        self, dataset: _HatecompDataset
    ) -> Tuple[
        torch.utils.data.DataLoader, torch.utils.data.DataLoader, List[torch.Tensor]
    ]:
        train_set, test_set = dataset.split(0.1)
        class_weights = [
            task_weights.to(self.configuration.device)
            for task_weights in get_class_weights(train_set, dataset.num_classes)
        ]
        train_dataloader = DataLoader(
            train_set,
            batch_size=self.configuration.training_batch_size,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            test_set,
            batch_size=self.configuration.evaluation_batch_size,
            shuffle=False,
        )
        return train_dataloader, test_dataloader, class_weights

    def get_tokenizer_and_model(self) -> Tuple[HatecompTokenizer, HatecompClassifier]:
        model = HatecompClassifier.from_huggingface_pretrained(
            self.configuration.transformer_name,
            self.dataset.num_classes,
            self.configuration.head_hidden_size,
            self.configuration.dropout,
        )
        tokenizer = HatecompTokenizer.from_huggingface_pretrained(
            self.configuration.transformer_name,
            model.transformer.config.max_position_embeddings - 2,
        )
        model.to(self.configuration.device)
        return tokenizer, model

    def get_optimizer_and_scheduler(
        self, model: HatecompClassifier
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.configuration.learning_rate,
            weight_decay=self.configuration.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.configuration.learning_rate,
            steps_per_epoch=len(self.training_dataloader),
            epochs=self.configuration.epochs,
            anneal_strategy="cos",
        )
        return optimizer, scheduler

    def get_loss_function(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if all([class_num == 1 for class_num in self.dataset.num_classes]):
            return lambda input, target, weight: torch.nn.functional.binary_cross_entropy_with_logits(
                input, target, pos_weight=weight
            )
        else:
            return torch.nn.functional.cross_entropy

    def run(self) -> float:
        trainer = HatecompTrainer(
            self.trial_dir,
            self.model,
            self.tokenizer,
            self.optimizer,
            self.scheduler,
            self.loss_function,
            self.training_dataloader,
            self.test_dataloader,
            self.configuration.epochs,
            self.class_weights,
            self.verbose,
            self.checkpoint,
        )
        trainer.train(self.configuration.device)
        # Get the avg macro F1 score from each epoch
        f1s = [epoch["test_f1s"] for epoch in trainer.metrics]
        macro_f1s = [[torch.mean(class_f1s) for class_f1s in epoch] for epoch in f1s]
        avg_macro_f1s = [torch.mean(torch.stack(epoch)).item() for epoch in macro_f1s]
        # Return the best macro F1 score

        # Potentially unnecessary cleanup
        del self.model
        del self.tokenizer
        del self.optimizer
        del self.scheduler
        del self.training_dataloader
        del self.test_dataloader
        del self.class_weights
        torch.cuda.empty_cache()

        return max(avg_macro_f1s)
