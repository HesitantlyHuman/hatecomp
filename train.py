import torch
import numpy as np
from hatecomp import NAACL
from hatecomp.base.utils import tokenize_bookends
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

raw_dataset = NAACL()

huggingface_model = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
tokenizer_function = lambda tokenization_input: tokenize_bookends(
    tokenization_input, 512, tokenizer
)
tokenized_dataset = raw_dataset.map(tokenizer_function, batched=True)
train_split, test_split = tokenized_dataset.split()

model = AutoModelForSequenceClassification.from_pretrained(
    huggingface_model, num_labels=1
)

accuracy = load_metric("accuracy")
f1_score = load_metric("f1_score")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return (
        accuracy.compute(predictions=predictions, references=labels),
        f1_score.compute(predictions=predictions, references=labels)
    )

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    evaluation_strategy = "epoch",
    weight_decay=0.01,
    dataloader_num_workers=8,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=test_split,
    compute_metrics=compute_metrics
)
trainer.train()
