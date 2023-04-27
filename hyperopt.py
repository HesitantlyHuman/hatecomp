import os
import gc
import json

import torch
import optuna

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

experiment_root = "experiments"
dataset = HASOC()
num_trials = 10

print("Loaded dataset: ", dataset.__name__)

experiment_directory = os.path.join(experiment_root, dataset.__name__)
os.makedirs(experiment_directory, exist_ok=True)

best_so_far = 0.0
best_so_far_location = None

distributions = {
    "transformer_name": optuna.distributions.CategoricalDistribution(
        [
            "roberta-base",
            "xlm-roberta-base",
            "distilroberta-base",
        ]
    ),
    "epochs": optuna.distributions.IntDistribution(3, 30),
    "training_batch_size": optuna.distributions.IntDistribution(8, 32),
    "learning_rate": optuna.distributions.FloatDistribution(1e-6, 1e-4, log=True),
    "weight_decay": optuna.distributions.FloatDistribution(1e-4, 0.01, log=True),
    "learning_rate_warmup_percentage": optuna.distributions.FloatDistribution(0.1, 0.4),
    "head_hidden_size": optuna.distributions.IntDistribution(256, 1024),
    "dropout": optuna.distributions.FloatDistribution(0.0001, 0.1, log=True),
}


def load_trial(directory: str) -> optuna.trial.Trial:
    if not os.path.exists(directory):
        return None
    trial_name = os.path.basename(directory)
    trial_parameters_path = os.path.join(directory, "trial_parameters.json")
    if not os.path.exists(trial_parameters_path):
        return None
    trial_metrics_path = os.path.join(directory, "metrics.json")
    if not os.path.exists(trial_metrics_path):
        return None
    with open(trial_parameters_path, "r") as f:
        trial_parameters = json.load(f)
    with open(trial_metrics_path, "r") as f:
        trial_metrics = json.load(f)
    f1s = [epoch["test_f1s"] for epoch in trial_metrics]
    f1s = [[sum(head) / len(head) for head in epoch] for epoch in f1s]
    f1s = [sum(epoch) / len(epoch) for epoch in f1s]
    best_f1 = max(f1s)
    trial = optuna.trial.create_trial(
        params={
            key: value
            for key, value in trial_parameters.items()
            if key not in ["name", "evaluation_batch_size", "device"]
        },
        distributions=distributions,
        value=best_f1,
        state=optuna.trial.TrialState.COMPLETE,
    )
    trial._trial_id = trial_name
    global best_so_far
    global best_so_far_location
    if best_f1 > best_so_far:
        best_so_far = best_f1
        best_so_far_location = os.path.join(directory, "best_model.pt")
    return trial


def load_trials(directory: str):
    if not os.path.exists(directory):
        return []
    trial_candidates = os.listdir(directory)
    trials = []
    for trial_candidate in trial_candidates:
        trial_directory = os.path.join(directory, trial_candidate)
        trial = load_trial(trial_directory)
        if trial is not None:
            trials.append(trial)
    print("Loaded ", len(trials), " trials from ", directory)
    global best_so_far
    global best_so_far_location
    print("Best so far: ", best_so_far, " at ", best_so_far_location)
    return trials


def objective(trial: optuna.trial.Trial):
    transformer_name = trial._suggest(
        "transformer_name", distributions["transformer_name"]
    )
    epochs = trial._suggest("epochs", distributions["epochs"])
    training_batch_size = trial._suggest(
        "training_batch_size", distributions["training_batch_size"]
    )
    learning_rate = trial._suggest("learning_rate", distributions["learning_rate"])
    weight_decay = trial._suggest("weight_decay", distributions["weight_decay"])
    learning_rate_warmup_percentage = trial._suggest(
        "learning_rate_warmup_percentage",
        distributions["learning_rate_warmup_percentage"],
    )
    dropout = trial._suggest("dropout", distributions["dropout"])
    head_hidden_size = trial._suggest(
        "head_hidden_size", distributions["head_hidden_size"]
    )

    config = HatecompTrialConfig(
        transformer_name=transformer_name,
        epochs=epochs,
        training_batch_size=training_batch_size,
        evaluation_batch_size=64,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        learning_rate_warmup_percentage=learning_rate_warmup_percentage,
        dropout=dropout,
        head_hidden_size=head_hidden_size,
    )
    trial_runner = HatecompTrialRunner(
        experiment_directory, dataset, config, checkpoint=True, verbose=False
    )
    print("Starting trial: ", config.name)
    result = trial_runner.run()
    del trial_runner
    gc.collect()
    torch.cuda.empty_cache()

    global best_so_far
    global best_so_far_location
    if result > best_so_far:
        if best_so_far_location is not None:
            os.remove(best_so_far_location)
        best_so_far = result
        best_so_far_location = os.path.join(
            experiment_directory, config.name, "best_model.pt"
        )
    else:
        # Delete the best model generated by this trial
        best_model_path = os.path.join(
            experiment_directory, config.name, "best_model.pt"
        )
        os.remove(best_model_path)

    return result


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.add_trials(load_trials(experiment_directory))
    study.optimize(objective, n_trials=num_trials)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    failed_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.FAIL]
    )

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("  Number of failed trials: ", len(failed_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
