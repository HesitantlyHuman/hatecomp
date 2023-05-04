from typing import List
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from hatecomp.datasets.base.utils import batch_and_slice
from hatecomp.models import HatecompClassifier, HatecompTokenizer
from hatespace.datasets import IronMarch
from hatecomp.datasets import (
    MLMA,
    Vicomtech,
    NAACL,
    NLPCSS,
    TwitterSexism,
    HASOC,
    WikiToxicity,
    WikiAggression,
    WikiPersonalAttacks,
)


def embed(
    data: List[str],
    model: HatecompClassifier,
    tokenizer: HatecompTokenizer,
    batch_size: int = 32,
    device: str = "cuda",
    progress_bar: bool = True,
) -> List[List[float]]:
    device = torch.device(device)

    # Check if the device is available
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available! using CPU instead...")
        device = torch.device("cpu")

    # Move the model to the device
    model.to(device)
    model.eval()

    embeddings = []

    # Create minibatches
    minibatches = batch_and_slice(data, batch_size)

    if progress_bar:
        from tqdm import tqdm

        total = np.ceil(len(data) / batch_size)
        minibatches = tqdm(minibatches, total=total)

    with torch.no_grad():
        for _, minibatch in minibatches:
            tokenized = tokenizer(minibatch)
            tokenized = {key: value.to(device) for key, value in tokenized.items()}

            minibatch_outputs = model(**tokenized)

            batch_embeddings = minibatch_outputs[0].cpu()
            for minibatch_output in minibatch_outputs[1:]:
                batch_embeddings = torch.cat(
                    (batch_embeddings, minibatch_output.cpu()),
                    dim=1,
                )

            embeddings.extend(batch_embeddings.tolist())

    return embeddings


def generate_kde_distributions(output_path: str, embeddings: List[List[float]]):
    # Use the KDE algorithm to generate distributions
    # Plot the distributions using matplotlib
    os.makedirs(output_path, exist_ok=True)

    embedding_length = len(embeddings[0])
    for i in range(embedding_length):
        # Get the ith dimension of the embeddings
        dimension = [embedding[i] for embedding in embeddings]

        # Generate the distribution
        distribution = gaussian_kde(dimension)

        # Plot the distribution
        x = np.linspace(min(dimension), max(dimension), 1000)
        plt.plot(x, distribution(x))

        # Add shading below the distribution
        plt.fill_between(x, distribution(x), alpha=0.2)

        plt.savefig(os.path.join(output_path, f"dimension_{i}.png"))
        plt.clf()


def generate_comparative_kde_distributions(
    output_path: str,
    embeddings_one: List[List[float]],
    embeddings_two: List[List[float]],
    embeddings_one_label: str = "Embeddings One",
    embeddings_two_label: str = "Embeddings Two",
    figure_title: str = "Comparative KDE Distributions",
):
    # Use the KDE algorithm to generate distributions
    # Plot the distributions using matplotlib
    os.makedirs(output_path, exist_ok=True)

    # Set colors
    color_one = "blue"
    color_two = "orange"

    embedding_length = len(embeddings_one[0])

    single_plot_size = 4

    if embedding_length < 4:
        single_plot_size = 6
        fig, axes = plt.subplots(
            1,
            embedding_length,
            figsize=(
                single_plot_size * embedding_length,
                single_plot_size,
            ),
            constrained_layout=True,
        )
    else:
        if embedding_length == 4:
            single_plot_size = 5
        next_square_edge = int(embedding_length**0.5 + 0.5)
        fig, axes = plt.subplots(
            next_square_edge,
            next_square_edge,
            figsize=(
                single_plot_size * next_square_edge,
                single_plot_size * next_square_edge,
            ),
            constrained_layout=True,
        )

    for i in range(embedding_length):
        if embedding_length == 1:
            ax = axes
        elif embedding_length < 4:
            ax = axes[i]
        else:
            axes_x, axes_y = i // next_square_edge, i % next_square_edge
            ax = axes[axes_x, axes_y]

        # Get the ith dimension of the embeddings
        dimension_one = [embedding[i] for embedding in embeddings_one]
        dimension_two = [embedding[i] for embedding in embeddings_two]

        # Generate the distribution
        distribution_one = gaussian_kde(dimension_one)
        distribution_two = gaussian_kde(dimension_two)

        x = np.linspace(
            min(dimension_one + dimension_two), max(dimension_one + dimension_two), 1000
        )

        distribution_one = distribution_one(x)
        distribution_two = distribution_two(x)

        # Plot the distribution
        ax.plot(x, distribution_one, label=embeddings_one_label, color=color_one)
        ax.plot(x, distribution_two, label=embeddings_two_label, color=color_two)

        # Add shading below the distribution
        ax.fill_between(x, distribution_one, alpha=0.2, color=color_one)
        ax.fill_between(x, distribution_two, alpha=0.2, color=color_two)

        ax.set_title(f"Dimension {i}")

    # Remove empty plots
    if embedding_length >= 4:
        for i in range(embedding_length, next_square_edge**2):
            axes_x, axes_y = i // next_square_edge, i % next_square_edge
            ax = axes[axes_x, axes_y]
            ax.axis("off")

    fig.legend(
        handles=[
            mpatches.Patch(color=color_one, label=embeddings_one_label),
            mpatches.Patch(color=color_two, label=embeddings_two_label),
        ],
        loc="lower right",
        ncol=2,
        borderaxespad=1,
    )

    fig.suptitle(figure_title, fontsize=24, wrap=True)

    fig.savefig(os.path.join(output_path, f"per_dim_distributions.png"))


if __name__ == "__main__":
    datasets = {
        # "HASOC": HASOC,
        # "MLMA": MLMA,
        # "Vicomtech": Vicomtech,
        # "NAACL": NAACL,
        # "NLPCSS": NLPCSS,
        # "TwitterSexism": TwitterSexism,
        # "WikiToxicity": WikiToxicity,
        "WikiAggression": WikiAggression,
        "WikiPersonalAttacks": WikiPersonalAttacks,
    }

    for dataset_name, dataset in datasets.items():
        print(f"Generating KDE distributions for {dataset_name}")
        tokenizer = HatecompTokenizer.from_hatecomp_pretrained(
            dataset_name, download=True
        )
        model = HatecompClassifier.from_hatecomp_pretrained(dataset_name)

        dataset_one = IronMarch("data/iron_march_201911")
        data = [item["data"] for item in dataset_one]
        embeddings_one = embed(data, model, tokenizer)

        dataset_two = dataset()
        data = [item["data"] for item in dataset_two]
        embeddings_two = embed(data, model, tokenizer)

        generate_comparative_kde_distributions(
            f"figures/{dataset_name}",
            embeddings_one,
            embeddings_two,
            "Iron March",
            dataset_name,
            f"Comparative KDE Distributions over Side-Information Features ({dataset_name} vs Iron March)",
        )
