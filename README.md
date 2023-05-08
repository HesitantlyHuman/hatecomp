# Hate Datasets Compilation
Contains 17 different hate, violence, and discrimination speech datasets, along with anotations on where they were found, the data format and method for collection and labeling. Each dataset is kept in the original file structure and placed inside the respective folder's `data` file, in case a more recent version is obtained. (Links for the source are in each ABOUT.md file)

## Installing
To use the hatecomp datasets or models, simply run the following command in a python environment of your choice:
```shell
pip install hatecomp
```

If you do not have pytorch already installed, it is recommended to do so using conda. Visit the pytorch [website](https://pytorch.org/) for more information.

Once it has finished downloading, you can start loading in datasets and models. Below are a couple of examples to get you started. For more advanced usage, please see the `train.py` file in the root of this repo. `hatecomp` follows the huggingface training API, so most everything that works with huggingface will work here.

## Examples
Here are a couple examples of how to use the hatecomp library.

### Working with a Dataset
Loading datasets is very simple. Each has its own downloading script that will run lazily when you try to create the dataset. If you would like to, you can specify where to download and if the dataset should download. By default the datasets only download when they cannot find the necessary files in the given location.
```python
from hatecomp.datasets import Vicomtech

# load a dataset from the default location,
# or download the dataset in the default location
dataset = Vicomtech()
example = dataset[0]

# load a dataset from a specified location,
# or download to that location
dataset = Vicomtech(root = "my/special/dataset/path")
example = dataset[0]

# only load a dataset if it can be found at the given location
dataset = Vicomtech(root = "my/special/dataset/path", download = False)
example = dataset[0]
```

The datasets also come equipped with a couple of handy features designed especially for NLP use and convenience.

```python
from hatecomp.datasets import Vicomtech

# Mapping a function over the dataset data (usually text, unless the dataset has already been mapped)
# Note that the map function can support batching if your mapped function supports it.
def my_tokenizing_function(some_string):
    return 0
dataset = Vicomtech()
tokenized_dataset = dataset.map(function = my_tokenizing_function, batched = False)

# Splitting the dataset
train_split, test_split = tokenized_dataset.spit(test_proportion = 0.1)
```

### Using a model
Hatecomp also provides functionality for training models with these datasets, along with some pretrained models.

#### Importing a pretrained model
Loading one of our pretrained models is quite simple, and only requires using the name of the appropriate dataset to do so. The model will then be downloaded into the files of the local `hatecomp` package. Note this means that uninstalling `hatecomp` will delete the models, and this is intended behavior.
```python
from hatecomp.models import HatecompClassifier, HatecompTokenizer

# To load an already downloaded model
model = HatecompClassifier.from_hatecomp_pretrained("Vicomtech")
tokenizer = HatecompTokenizer.from_hatecomp_pretrained("Vicomtech")

# To download a model if it does not exist locally
model = HatecompClassifier.from_hatecomp_pretrained("Vicomtech", download=True)
tokenizer = HatecompTokenizer.from_hatecomp_pretrained("Vicomtech")

# Force download a model (useful if the files become corrupted for any reason)
model = HatecompClassifier.from_hatecomp_pretrained("Vicomtech", force_download=True)
tokenizer = HatecompTokenizer.from_hatecomp_pretrained("Vicomtech")
```
The tokenizers also have the same `download` and `force_download` flags available, but if you are loading the tokenizer directly after the model, the files will be installed locally already, as the download retrieves the necessary data for both the model and the tokenizer.

#### Training models
The process for training a model is quite simple, as there is a custom trainer class designed specifically for the datasets and models. Also included is a convenience `DataLoader` wrapper, which will handle collating the hatecomp data, since hatecomp datasets return ids, and the base `torch.utils.data.DataLoader` does not handle those by default. 
```python
import torch

from hatecomp.datasets import Vicomtech
from hatecomp.datasets.base import DataLoader

from hatecomp.training import HatecompTrainer
from hatecomp.models import HatecompClassifier, HatecompTokenizer

dataset = Vicomtech()
model = HatecompClassifier.from_huggingface_pretrained(
    "roberta-base",
    dataset.num_classes
)
tokenizer = HatecompTokenizer.from_huggingface_pretrained(
    "roberta-base",
)

train_set, test_set = dataset.split(0.1)
train_dataloader = DataLoader(train_set, batch_size=32)
test_dataloader = DataLoader(test_set, batch_size=64)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-5
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-5,
    steps_per_epoch=len(train_dataloader),
    epochs=5
)

loss_function = torch.nn.functional.cross_entropy

trainer = HatecompTrainer(
    root="root_directory",
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_function=loss_function,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    epochs=5
)
trainer.train("cuda")
```

For an even more expedited training process, there is also the `HatecompTrialRunner`, an example of using this class can be found in `scripts/hyperopt.py`.

## Datasets
Additional information about each dataset can be found in the corresponding ABOUT.md file. 

### Functionality
Currently the following datasets are implemented:
- [ZeerakTalat NAACL](hatecomp/datasets/ZeerakTalat/README.md)
- [ZeerakTalat NLPCSS](hatecomp/datasets/ZeerakTalat/README.md)
- [HASOC](hatecomp/datasets/HASOC/README.md)
- [Vicomtech](hatecomp/datasets/Vicomtech/README.md)
- [TwitterSexism](hatecomp/datasets/TwitterSexism/README.md)
- [MLMA](hatecomp/datasets/MLMA/README.md)

Several more have downloaders already, and are close to completion

### Notes
Two of the dataset, the `MLMA Dataset` and `Online Intervention Dataset`, only contain hateful posts, instead labeling other features such as the target of hate.

## Training
`hatecomp` provides some basic training tools to integrate into the [huggingface](https://github.com/huggingface) :hugs: Trainer API. A full example of how to train a model using the hatecomp datasets can be found in `train.py`

### Results
Here is a list of results acheived on various datasets with Huggingface models, along with the SOTA performance (as best I could find). Since it is not always possible to find SOTA scores for obscure datsets measured with a particular metric, the hatecomp score is selected to match whatever SOTA could be found. (The links are locations where the SOTA reference was found. If you are looking for citations, please refer to the `About.md` for each dataset)

| Dataset | Metric | SOTA | hatecomp/huggingface |
| -- | -- | -- | -- |
| [Vicomtech](https://arxiv.org/pdf/1809.04444.pdf) | Accuracy | 0.79 | **0.93** |
| [ZeerakTalat-NAACL](https://aclanthology.org/N16-2013.pdf) | F1 | 0.74 | **0.94** |
| [ZeerakTalat-NLPCSS](https://aclanthology.org/W16-5618.pdf) | F1 | 0.53 | **0.76** |
| [HASOC](https://arxiv.org/pdf/2108.05927.pdf) | F1 (Macro Average) | 0.53 | **0.55** |
| [TwitterSexism](https://aclanthology.org/W17-2902.pdf) | F1 (Macro) | 0.87 | **0.99** |
| [MLMA](https://arxiv.org/pdf/1908.11049.pdf) | F1 (Multitask Macros EN) | [0.30, 0.43, **0.18**, **0.57**] | [**0.58**, **0.64**, 0.16, 0.51] |

(If you know of a better SOTA than what is listed here, please create an issue or pull request.)

Also note that some of these datasets require tweet data. For these, a large number of tweet_ids return Unauthorized from the twitter API, so the data which the hatecomp models trained on is a subset of the total dataset. More information can be found in the following table:

| Dataset | Total Size | Successfully Downloaded Tweets | Available Training Portion |
| -- | -- | -- | -- |
| ZeerakTalat-NAACL | 16907 | 7210 | 0.4264 |
| ZeerakTalat-NLPCSS | 6909 | 5385 | 0.77941 |
| TwitterSexism | 10583 | 5054 | 0.4775 |

This info is valid as of Feb 2022, and is probably subject to change as Twitter continues to lock down their API.

## TODO
Better names for some of these datasets. Probably don't want a dataset named after a conference or github handle
Update README