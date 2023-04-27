import os
import json
from urllib.request import urlretrieve

from hatecomp._path import install_path


FILE_EXTENSIONS = {
    "model": "model.pt",
    "config": "config.json",
}

PRETRAINED_INSTALLATION_LOCATION = os.path.join(install_path, "models", "pretrained")

# Load the download json file
model_registry_path = os.path.join(install_path, "models", "model_registry.json")
with open(model_registry_path, "r") as f:
    MODEL_REGISTRY = json.load(f)
    MODEL_REGISTRY = {k.lower(): v for k, v in MODEL_REGISTRY.items()}

def verify_pretrained_download(
    pretrained_model_name_or_path: str,
    download: bool = False,
    force_download: bool = False
) -> None:
    pretrained_model_name_or_path = pretrained_model_name_or_path.lower()
    local_path = os.path.join(
        PRETRAINED_INSTALLATION_LOCATION, pretrained_model_name_or_path
    )
    if not os.path.exists(local_path) or force_download:
        if download or force_download:
            download_model(pretrained_model_name_or_path)
        else:
            raise FileNotFoundError(
                f"Could not find the model {pretrained_model_name_or_path} in the local directory. "
                f"If you want to download the model, set download=True."
            )

def download_model(model_name: str, verbose: bool = True) -> None:
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Could not find {model_name} in the model registry.")

    for (file_type, url) in MODEL_REGISTRY[model_name].items():
        progress_bar = lambda b, bsize, tsize: None
        if verbose:
            try:
                from tqdm import tqdm

                class DownloadProgressBar:
                    def __init__(self, **kwargs) -> None:
                        self.bar = tqdm(**kwargs)

                    def __call__(
                        self, b: int = 1, bsize: int = 1, tsize: int = None
                    ) -> None:
                        if tsize is not None:
                            self.bar.total = tsize
                        self.bar.update(b * bsize - self.bar.n)

                progress_bar = DownloadProgressBar(total=0, unit="B", unit_scale=True)

            except ImportError:
                raise ImportError(
                    "Please install tqdm to use the download function with verbose=True."
                )

        file_name = os.path.join(
            PRETRAINED_INSTALLATION_LOCATION, model_name, FILE_EXTENSIONS[file_type]
        )

        # Delete the file if it already exists
        if os.path.exists(file_name):
            os.remove(file_name)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        # Download the file
        urlretrieve(url, file_name, progress_bar)


if __name__ == "__main__":
    download_model("MLMA")
