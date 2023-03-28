import os

from hatecomp.datasets import MLMA, HASOC, NAACL, NLPCSS, Vicomtech, TwitterSexism
from hatecomp.training import HatecompTrialRunner, HatecompTrialConfig

checkpoint_dir = "experiments"
dataset = TwitterSexism()
root = os.path.join(checkpoint_dir, dataset.__name__)

print(dataset.num_classes)
asdf

config = HatecompTrialConfig(
    transformer_name="roberta-base",
    epochs=5,
    training_batch_size=16,
    evaluation_batch_size=64,
    learning_rate=0.9e-5,
    weight_decay=0.01,
    learning_rate_warmup_percentage=0.3,
    dropout=0.01,
    head_hidden_size=768,
    device="cuda",
)
trial_runner = HatecompTrialRunner(root, dataset, config, checkpoint=True)
trial_runner.run()
