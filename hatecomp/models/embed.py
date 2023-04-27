from typing import List
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from hatecomp.datasets.base.utils import batch_and_slice
from hatecomp.models import HatecompClassifier, HatecompTokenizer
from hatespace.datasets import IronMarch
from hatecomp.datasets import MLMA, Vicomtech, NAACL, NLPCSS, TwitterSexism

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
            tokenized = {
                key: value.to(device)
                for key, value in tokenized.items()
            }

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
    ):
    # Use the KDE algorithm to generate distributions
    # Plot the distributions using matplotlib
    os.makedirs(output_path, exist_ok=True)

    # Set colors
    color_one = "blue"
    color_two = "orange"

    embedding_length = len(embeddings_one[0])
    for i in range(embedding_length):
        # Get the ith dimension of the embeddings
        dimension_one = [embedding[i] for embedding in embeddings_one]
        dimension_two = [embedding[i] for embedding in embeddings_two]

        # Generate the distribution
        distribution_one = gaussian_kde(dimension_one)
        distribution_two = gaussian_kde(dimension_two)

        # Plot the distribution
        x = np.linspace(min(dimension_one + dimension_two), max(dimension_one + dimension_two), 1000)
        plt.plot(x, distribution_one(x), label=embeddings_one_label, color=color_one)
        plt.plot(x, distribution_two(x), label=embeddings_two_label, color=color_two)

        # Add shading below the distribution
        plt.fill_between(x, distribution_one(x), alpha=0.2, color=color_one)
        plt.fill_between(x, distribution_two(x), alpha=0.2, color=color_two)

        plt.legend()
        plt.savefig(os.path.join(output_path, f"dimension_{i}.png"))
        plt.clf()


if __name__ == "__main__":
    dataset_name = "MLMA"
    dataset = MLMA

    tokenizer = HatecompTokenizer.from_hatecomp_pretrained(dataset_name, download=True)
    tokenizer._start_token_length = int(512 / 2)
    tokenizer._end_token_length = 512 - tokenizer._start_token_length
    tokenizer.max_length = 512
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
    )