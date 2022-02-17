import numpy as np
from hatecomp import NAACL, Vicomtech
from hatecomp.base.training import HatecompTrainer
from hatecomp.base.utils import tokenize_bookends
from hatecomp.base.metrics import Accuracy, F1
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments

raw_dataset = Vicomtech()

huggingface_model = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
tokenizer_function = lambda tokenization_input: tokenize_bookends(
    tokenization_input, 512, tokenizer
)
tokenized_dataset = raw_dataset.map(tokenizer_function, batched=True)
train_split, test_split = tokenized_dataset.split()

num_classes = tokenized_dataset.num_classes()
model = AutoModelForSequenceClassification.from_pretrained(
    huggingface_model, num_labels=num_classes
)

metrics = [Accuracy(), F1(num_classes)]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_outputs = {}
    for metric in metrics:
        metric_outputs.update(
            metric.compute(predictions=predictions, references=labels)
        )
    return metric_outputs


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    weight_decay=0.1,
    dataloader_num_workers=8,
)
trainer = HatecompTrainer(
    model=model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=test_split,
    compute_metrics=compute_metrics,
)
trainer.train()
