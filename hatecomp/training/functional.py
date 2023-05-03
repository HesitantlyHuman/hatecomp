from typing import Dict, Union, Callable, List, Tuple
import torch
import shutil
from tqdm import tqdm

EPS = 1e-15


def print_centered(message):
    terminal_size = shutil.get_terminal_size()
    padding = (terminal_size.columns - len(message)) // 2
    print(" " * padding + message)


def tokenize_batch(
    tokenizer: Callable[[str], Dict[str, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    device: Union[str, torch.device],
) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(batch["data"])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def split_label_tensor(
    batch: Dict[str, torch.Tensor], device: Union[str, torch.device]
) -> Dict[str, torch.Tensor]:
    labels = batch["labels"]
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels = torch.split(batch["labels"], 1, dim=1)
        labels = [label.squeeze(1) for label in labels]
    else:
        labels = [labels]
    labels = [label.to(device) for label in labels]
    return labels


def compute_loss(
    model: torch.nn.Module,
    tokenizer: Callable[[str], Dict[str, torch.Tensor]],
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    class_weights: List[torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: Union[str, torch.device],
    return_outputs=False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    inputs = tokenize_batch(tokenizer, batch, device)
    labels = split_label_tensor(batch, device)

    outputs = model(**inputs)

    loss = torch.tensor(0.0).to("cuda:0")
    for head_idx, head_outputs in enumerate(outputs):
        loss = loss_function(
            torch.squeeze(head_outputs, dim=1),
            labels[head_idx],
            weight=class_weights[head_idx],
        )

    return (loss, outputs) if return_outputs else loss


def split_batch_into_minibatches(
    batch: Dict[str, torch.Tensor], minibatch_size: int
) -> Dict[str, torch.Tensor]:
    minibatches = []
    for i in range(0, len(batch["data"]), minibatch_size):
        minibatch = {key: value[i : i + minibatch_size] for key, value in batch.items()}
        minibatches.append(minibatch)
    return minibatches


def training_step(
    model: torch.nn.Module,
    tokenizer: Callable[[str], Dict[str, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    class_weights: List[torch.Tensor],
    device: Union[str, torch.device],
    minibatch_size: int = 8,
) -> torch.Tensor:
    minibatches = split_batch_into_minibatches(batch, minibatch_size)

    def training_ministep(minibatch) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(
            model, tokenizer, loss_function, class_weights, minibatch, device
        )
        loss.backward()
        return loss.detach()

    loss = torch.tensor(0.0).to(device)
    for minibatch in minibatches:
        loss += training_ministep(minibatch)

    optimizer.step()
    scheduler.step()

    return loss / len(minibatches)


def training_epoch(
    model: torch.nn.Module,
    tokenizer: Callable[[str], Dict[str, torch.Tensor]],
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    class_weights: List[torch.Tensor],
    device: Union[str, torch.device],
    minibatch_size: int = 8,
    running_average_length: int = 30,
    verbose: bool = True,
) -> torch.Tensor:
    model.train()
    if verbose:
        dataloader = tqdm(dataloader, desc="Training")
    losses = []
    for batch in dataloader:
        loss = training_step(
            model,
            tokenizer,
            batch,
            optimizer,
            scheduler,
            loss_function,
            class_weights,
            device,
            minibatch_size,
        )
        losses.append(loss.item())
        if verbose:
            dataloader.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{sum(losses[-running_average_length:])/len(losses[-running_average_length:]):.4f}",
            )
    return sum(losses) / len(losses)


def calculate_confusion_matrices(
    batch: Dict[str, torch.Tensor],
    outputs: List[torch.Tensor],
    device: Union[str, torch.device] = "cpu",
) -> List[torch.Tensor]:
    labels = split_label_tensor(batch, device)
    confusion_matrices = []
    for head_idx, head_outputs in enumerate(outputs):
        if not labels[head_idx].dtype in [torch.long, torch.int, torch.bool]:
            thresholded_labels, head_outputs = (
                (labels[head_idx] > 0.5).long(),
                torch.squeeze(head_outputs > 0.0).long(),
            )
            confusion_matrix = torch.zeros((2, 2), dtype=torch.long).to(device)
            for i in range(len(head_outputs)):
                confusion_matrix[head_outputs[i], thresholded_labels[i]] += 1
            confusion_matrices.append(confusion_matrix)
        else:
            num_classes = head_outputs.shape[1]
            head_outputs = torch.argmax(head_outputs, dim=1)
            confusion_matrix = torch.zeros(
                (num_classes, num_classes), dtype=torch.long
            ).to(device)
            for i in range(len(head_outputs)):
                confusion_matrix[head_outputs[i], labels[head_idx][i]] += 1
            confusion_matrices.append(confusion_matrix)
    return confusion_matrices


def evaluation_step(
    model: torch.nn.Module,
    tokenizer: Callable[[str], Dict[str, torch.Tensor]],
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    class_weights: List[torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: Union[str, torch.device],
    minibatch_size: int = 64,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    minibatches = split_batch_into_minibatches(batch, minibatch_size)

    def evaluation_ministep(minibatch) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        loss, outputs = compute_loss(
            model,
            tokenizer,
            loss_function,
            class_weights,
            minibatch,
            device,
            return_outputs=True,
        )
        confusion_matrices = calculate_confusion_matrices(minibatch, outputs, device)
        return loss, confusion_matrices

    loss = torch.tensor(0.0).to(device)
    confusion_sum = None
    for minibatch in minibatches:
        loss, confusion_matrices = evaluation_ministep(minibatch)

        loss += loss

        if confusion_matrices is None:
            continue

        if confusion_sum is None:
            confusion_sum = confusion_matrices
        else:
            confusion_sum += confusion_matrices

    return loss / len(minibatches), confusion_sum


def f1_score(confusion_matrix: torch.Tensor) -> float:
    # Calculate the F1 score for each class
    true_positives = confusion_matrix.diag()
    false_positives = confusion_matrix.sum(dim=0) - true_positives
    false_negatives = confusion_matrix.sum(dim=1) - true_positives
    f1_scores = (2 * true_positives + EPS) / (
        2 * true_positives + false_positives + false_negatives + EPS
    )
    return f1_scores


def accuracy(confusion_matrix: torch.Tensor) -> float:
    true_positives = confusion_matrix.diag()
    return true_positives.sum().item() / confusion_matrix.sum().item()


def evaluation_epoch(
    model: torch.nn.Module,
    tokenizer: Callable[[str], Dict[str, torch.Tensor]],
    dataloader: torch.utils.data.DataLoader,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    class_weights: List[torch.Tensor],
    device: Union[str, torch.device],
    minibatch_size: int = 64,
    running_average_length: int = 30,
    verbose: bool = True,
) -> Dict[str, float]:
    model.eval()
    if verbose:
        dataloader = tqdm(dataloader, desc="Evaluation")
    losses = []
    confusion_matrices = None
    with torch.no_grad():
        for batch in dataloader:
            loss, minibatch_confusion_matrices = evaluation_step(
                model,
                tokenizer,
                loss_function,
                class_weights,
                batch,
                device,
                minibatch_size,
            )
            losses.append(loss.item())
            if confusion_matrices is None:
                confusion_matrices = minibatch_confusion_matrices
            else:
                for i in range(len(confusion_matrices)):
                    confusion_matrices[i] += minibatch_confusion_matrices[i]

            # Calculate loss, macro f1 and accuracy for the last running_average_length batches
            running_loss = sum(losses[-running_average_length:]) / len(
                losses[-running_average_length:]
            )
            f1_scores = [
                f1_score(confusion_matrix) for confusion_matrix in confusion_matrices
            ]
            macro_f1s = [f1_score.mean().item() for f1_score in f1_scores]
            avg_macro_f1 = sum(macro_f1s) / len(macro_f1s)
            accuracies = [
                accuracy(confusion_matrix) for confusion_matrix in confusion_matrices
            ]
            avg_accuracy = sum(accuracies) / len(accuracies)

            if verbose:
                dataloader.set_postfix(
                    loss=f"{running_loss:.4f}",
                    macro_f1=f"{avg_macro_f1:.4f}",
                    accuracy=f"{avg_accuracy:.4f}",
                )

    # Calculate loss, f1 and accuracies for the whole epoch
    loss = sum(losses) / len(losses)
    f1s = [f1_score(confusion_matrix) for confusion_matrix in confusion_matrices]
    accuracies = [accuracy(confusion_matrix) for confusion_matrix in confusion_matrices]

    return {
        "loss": loss,
        "f1s": f1s,
        "accuracies": accuracies,
        "confusion_matrices": confusion_matrices,
    }


if __name__ == "__main__":
    # create a dummy confusion matrix and test the f1 and accuracy functions

    confusion_matrix = torch.tensor(
        [
            [210, 401, 44],
            [93, 123, 12],
            [12, 23, 34],
        ]
    )

    print(f1_score(confusion_matrix))
    print(accuracy(confusion_matrix))
