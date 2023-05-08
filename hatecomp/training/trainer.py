from typing import Callable, List
import os
import json
import torch

from hatecomp.training.functional import (
    print_centered,
    training_epoch,
    evaluation_epoch,
)

CHECKPOINT_FILENAME = "checkpoint.pt"
BEST_MODEL_FILENAME = "best_model.pt"
METRICS_FILENAME = "metrics.json"


class HatecompTrainer:
    def __init__(
        self,
        root: str,
        model: torch.nn.Module,
        tokenizer: Callable[[List[str]], List[int]],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        epochs: int,
        class_weights: torch.Tensor = None,
        verbose: bool = True,
        checkpoint: bool = True,
    ) -> None:
        self.root = root
        self.metrics = []
        self.best_loss = float("inf")
        self.epoch = 0
        self.epochs = epochs
        self.verbose = verbose

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        if class_weights is None:
            self.class_weights = [None for _ in range(len(model.heads))]
        else:
            self.class_weights = class_weights

        self.checkpoint = checkpoint
        if self.checkpoint:
            self.load_checkpoint()

    def load_checkpoint(self) -> None:
        # Verify that the root directory exists
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # Load the checkpoint if it exists
        checkpoint_path = os.path.join(self.root, CHECKPOINT_FILENAME)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.best_loss = checkpoint["best_loss"]
            self.log(
                f"Loaded checkpoint from epoch {self.epoch} with best loss {self.best_loss}"
            )

        # Load the metrics if they exist and convert the confusion matrices to tensors
        metrics_path = os.path.join(self.root, METRICS_FILENAME)
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as metrics_file:
                self.metrics = json.load(metrics_file)
            for epoch_metrics in self.metrics:
                epoch_metrics["test_confusion_matrices"] = [
                    torch.tensor(confusion_matrix)
                    for confusion_matrix in epoch_metrics["test_confusion_matrices"]
                ]

    def save_metrics(self) -> None:
        # Save the metrics, making sure that we convert the confusion matrices to lists
        # so that they can be serialized.
        with open(os.path.join(self.root, METRICS_FILENAME), "w") as f:
            formatted_metrics = [
                {
                    "epoch": metric["epoch"],
                    "training_loss": metric["training_loss"],
                    "test_loss": metric["test_loss"],
                    "test_f1s": [f1.tolist() for f1 in metric["test_f1s"]],
                    "test_accuracies": metric["test_accuracies"],
                    "test_confusion_matrices": [
                        confusion_matrix.tolist()
                        for confusion_matrix in metric["test_confusion_matrices"]
                    ],
                }
                for metric in self.metrics
            ]
            json.dump(formatted_metrics, f, indent=4)

    def save_checkpoint(self, epoch_loss: float) -> None:
        # Save the best model
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            torch.save(
                self.model.state_dict(), os.path.join(self.root, BEST_MODEL_FILENAME)
            )

        # Save the checkpoint
        torch.save(
            {
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            os.path.join(self.root, CHECKPOINT_FILENAME),
        )

    def delete_checkpoint(self) -> None:
        checkpoint_path = os.path.join(self.root, CHECKPOINT_FILENAME)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    def report_training_results(self) -> None:
        if self.checkpoint:
            self.log(
                f"Best model saved to '{os.path.abspath(os.path.join(self.root, BEST_MODEL_FILENAME))}'"
            )
        self.log(
            f"Metrics saved to '{os.path.abspath(os.path.join(self.root, METRICS_FILENAME))}'"
        )
        self.log("--- Metrics ---")
        best_epoch = min(self.metrics, key=lambda metric: metric["test_loss"])
        self.log(f"    Best epoch : {best_epoch['epoch'] + 1}")
        self.log(f"    Test loss : {best_epoch['test_loss']:.4f}")
        self.log("    Test Accuracies: ", end="")
        for metric in best_epoch["test_accuracies"]:
            self.log(f"{metric:.4f}, ", end="")
        self.log("")
        self.log("    Test F1s:", end="")
        for metric in best_epoch["test_f1s"]:
            self.log("\n        ", end="")
            for element in metric:
                self.log(f"{element:.4f}, ", end="")
        self.log("")

    def train(self, device: str = "cuda") -> None:
        self.model.to(device)
        self.log("--- Training ---", centered=True)
        for epoch in range(self.epoch, self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_metrics = {"epoch": epoch}
            training_loss = training_epoch(
                model=self.model,
                tokenizer=self.tokenizer,
                dataloader=self.train_dataloader,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                loss_function=self.loss_function,
                class_weights=self.class_weights,
                device=device,
            )
            epoch_metrics["training_loss"] = training_loss
            evaluation_metrics = evaluation_epoch(
                model=self.model,
                tokenizer=self.tokenizer,
                dataloader=self.test_dataloader,
                loss_function=self.loss_function,
                class_weights=self.class_weights,
                device=device,
            )
            epoch_metrics["test_loss"] = evaluation_metrics["loss"]
            epoch_metrics["test_f1s"] = evaluation_metrics["f1s"]
            epoch_metrics["test_accuracies"] = evaluation_metrics["accuracies"]
            epoch_metrics["test_confusion_matrices"] = evaluation_metrics[
                "confusion_matrices"
            ]
            self.epoch += 1
            self.metrics.append(epoch_metrics)
            if self.checkpoint:
                self.save_checkpoint(evaluation_metrics["loss"])
            self.save_metrics()
        self.delete_checkpoint()
        self.log("--- Training complete ---", centered=True)
        self.report_training_results()

        # Release the GPU memory manually because huggingface won't
        # clean up after itself.
        del self.model
        del self.optimizer
        del self.scheduler
        del self.class_weights
        torch.cuda.empty_cache()

    def log(self, string, centered: bool = False, **kwargs) -> None:
        if self.verbose:
            if centered:
                print_centered(string, **kwargs)
            else:
                print(string, **kwargs)
