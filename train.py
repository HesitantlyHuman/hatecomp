import os

from hatecomp.datasets import (
    MLMA,
    HASOC,
    NAACL,
    NLPCSS,
    Vicomtech,
    TwitterSexism,
    WikiToxicity,
    WikiAggression,
    WikiPersonalAttacks,
)
from hatecomp.training import HatecompTrialRunner, HatecompTrialConfig

checkpoint_dir = "experiments"
dataset = WikiPersonalAttacks()
root = os.path.join(checkpoint_dir, dataset.__name__)

config = HatecompTrialConfig(
    transformer_name="roberta-base",
    epochs=7,
    training_batch_size=16,
    evaluation_batch_size=64,
    learning_rate=0.5e-5,
    weight_decay=0.001,
    learning_rate_warmup_percentage=0.3,
    dropout=0.001,
    head_hidden_size=550,
    device="cuda",
)
trial_runner = HatecompTrialRunner(root, dataset, config, checkpoint=True)
trial_runner.run()
