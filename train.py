import torch
from hatecomp import NAACL
from hatecomp.base.utils import tokenize_bookends
from hatecomp.base import DataLoader
from hatecomp.base.training import HatecompTrainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments

raw_dataset = NAACL()
assert isinstance(raw_dataset, torch.utils.data.IterableDataset)

huggingface_model = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
tokenizer_function = lambda tokenization_input: tokenize_bookends(
    tokenization_input, 512, tokenizer
)
tokenized_dataset = raw_dataset.map(tokenizer_function, batched=True)
train_split, test_split = tokenized_dataset.split()

train_loader = DataLoader(train_split, batch_size=16, num_workers=8)
test_loader = DataLoader(test_split, batch_size=128, num_workers=8)

model = AutoModelForSequenceClassification.from_pretrained(
    huggingface_model, num_labels=1
)

training_args = TrainingArguments("test_trainer")
trainer = HatecompTrainer(
    model=model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=test_split,
)
trainer.train()
