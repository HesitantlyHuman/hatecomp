import numpy as np
from hatecomp.datasets import NAACL, NLPCSS, Vicomtech, TwitterSexism
from hatecomp.training import HatecompTrainer, HatecompTrainingArgs, Accuracy, F1
from hatecomp.models import AutoModelForSequenceClassification
from hatecomp.base.utils import tokenize_bookends
from transformers import AutoTokenizer

# Set the basic training parameters we will use in this run
training_config = {
    "epochs": 3,
    "train_batch_size": 16,
    "eval_batch_size": 128,
    "dataloader_workers": 12,
    "test_proportion": 0.3,
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "dropout": 0.2,  # Figure out how this is supposed to plumb through
    "warmup_ratio": 0.05,
    "lr_cycles": 2,
    "transformer_model": "roberta-base",
}

# Import a raw hatecomp dataset
raw_dataset = NAACL()
num_classes = raw_dataset.num_classes

# Use the huggingface auto classes to load a transformer and
# This will also configure the classification head for
# the correct number of classes.
tokenizer = AutoTokenizer.from_pretrained(training_config["transformer_model"])
model = AutoModelForSequenceClassification.from_pretrained(
    training_config["transformer_model"], num_labels=num_classes
)

# Create a tokenizer function that will tokenize to the model's max size
# using the hatecomp tokenize_bookends function. If the input text is too long
# it will grab half of its tokens from the beginning of the input, and half
# of its tokens from the end.
tokenizer_function = lambda tokenization_input: tokenize_bookends(
    tokenization_input, model.config.max_position_embeddings, tokenizer
)
tokenized_dataset = raw_dataset.map(tokenizer_function, batched=True)[:200]
train_split, test_split = tokenized_dataset.split(training_config["test_proportion"])

# Create a function to compute our accuracy and F1 metrics
metrics = [Accuracy(), F1(num_classes)]


def compute_metrics(eval_pred):
    print(eval_pred)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_outputs = {}
    for metric in metrics:
        metric_outputs.update(
            metric.compute(predictions=predictions, references=labels)
        )
    return metric_outputs


# Finally, create our training arguments, and the trainer itself,
# then start the training process. This will also checkpoint the model
# automatically every 500 steps.
training_args = HatecompTrainingArgs(
    output_dir="./results",
    learning_rate=training_config["learning_rate"],
    per_device_train_batch_size=training_config["train_batch_size"],
    per_device_eval_batch_size=training_config["eval_batch_size"],
    num_train_epochs=training_config["epochs"],
    evaluation_strategy="epoch",
    weight_decay=training_config["weight_decay"],
    dataloader_num_workers=training_config["dataloader_workers"],
    warmup_ratio=training_config["warmup_ratio"],
    lr_cycles=training_config["lr_cycles"],
)
trainer = HatecompTrainer(
    model=model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=test_split,
    compute_metrics=compute_metrics,
)
trainer.train()
