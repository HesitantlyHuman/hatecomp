import numpy as np


def calculate_confusion_matrix(predictions, references, num_classes):
    assert len(predictions) == len(
        references
    ), "Cannot calculate confusion matrix for logits and labels of different lengths"
    confusion_matrix = np.zeros((num_classes, num_classes))
    for pred, ref in zip(predictions, references):
        confusion_matrix[pred, ref] += 1
    return confusion_matrix


def calculate_class_f1(confusion_matrix, class_index):
    true_positive = confusion_matrix[class_index, class_index]
    false_positive = np.sum(confusion_matrix[class_index, :]) - true_positive
    false_negative = np.sum(confusion_matrix[:, class_index]) - true_positive
    precision = true_positive / (true_positive + false_positive + 1e-20)
    recall = true_positive / (true_positive + false_negative + 1e-20)
    return (2 * precision * recall) / (precision + recall + 1e-20)


class Accuracy:
    def compute(self, predictions, references):
        total = len(predictions)
        total_correct = np.sum(np.where(predictions == references, 1, 0))
        return {"accuracy": total_correct / total}


class F1:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute(self, predictions, references):
        f1_scores = []
        confusion_matrix = calculate_confusion_matrix(
            predictions, references, self.num_classes
        )
        for class_index in range(self.num_classes):
            f1_scores.append(calculate_class_f1(confusion_matrix, class_index))
        return {"f1": f1_scores}
