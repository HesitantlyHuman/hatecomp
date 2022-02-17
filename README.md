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
The process for using a model is very similar to the huggingface Trainer API.
```python
from hatecomp.datasets import Vicomtech
from hatecomp.training import Trainer, TrainingArguments
from hatecomp.base.utils import tokenize_bookends
from transformers import AutoTokenizer, AutoModelForSequenceClassification

raw_dataset = Vicomtech()
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
num_classes = raw_dataset.num_classes()
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels = num_classes
)

tokenizer_function = lambda tokenization_input: tokenize_bookends(
    tokenization_input, model.config.max_position_embeddings, tokenizer
)
tokenized_dataset = raw_dataset.map(tokenizer_function, batched=True)
train_split, test_split = tokenized_dataset.split()

training_args = TrainingArguments("test-run")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=test_split
)
trainer.train()
```

## Datasets
Additional information about each dataset can be found in the corresponding ABOUT.md file. 

### Functionality
Currently the following datasets are implemented:
- ZeerakTalat NAACL
- ZeerakTalat NLPCSS
- HASOC
- Vicomtech

Several more have downloaders already, and are close to completion

### Notes
Two of the dataset, the `MLMA Dataset` and `Online Intervention Dataset`, only contain hateful posts, instead labeling other features such as the target of hate.

## Training
`hatecomp` provides some basic training tools to integrate into the [huggingface](https://github.com/huggingface) :hugs: Trainer API. A full example of how to train a model using the hatecomp datasets can be found in `train.py`

### Results
Here is a list of results achieved on various datasets with Huggingface models, along with the SOTA performance (as best I could find). Some references only have accuracy, others only have F1. The hatecomp performance is measured with the appropriate metric for whatever SOTA could be found. (Dataset names link to the locations where each SOTA reference was found. If you are looking for citations, please refer to the `About.md` for each dataset)

| Dataset | Metric | SOTA | hatecomp/huggingface |
| -- | -- | -- | -- |
| [Vicomtech](https://arxiv.org/pdf/1809.04444.pdf) | Accuracy | 0.79 | **0.93** |
| [ZeerakTalat-NAACL](https://aclanthology.org/N16-2013.pdf) | F1 | 0.74 | **0.94** |
| [ZeerakTalat-NLPCSS](https://aclanthology.org/W16-5618.pdf) | F1 | 0.53 | NA |
| [HASOC](https://arxiv.org/pdf/2108.05927.pdf) | F1 (Macro) | 0.53 | NA |
| [TwitterSexism](https://aclanthology.org/W17-2902.pdf) | F1 (Macro) | 0.87 | **0.99** |

(If you know of a better SOTA than what is listed here, please create an issue or pull request.)

Also note that some of these datasets require tweet data. For these, a large number of tweet_ids return Unauthorized from the twitter API, so the data which the hatecomp models trained on is a subset of the total dataset. More information can be found in the following table:

| Dataset | Total Size | Successfully Downloaded Tweets | Available Training Portion |
| -- | -- | -- | -- |
| ZeerakTalat-NAACL | 16907 | 7210 | 0.4264 |
| ZeerakTalat-NLPCSS | 6909 | 4190 | 0.6064 |
| TwitterSexism | 10583 | 5054 | 0.4775 |

This info is valid as of Feb 2022, and is probably subject to change as Twitter continues to lock down their API.

## TODO
Implement a multi-task model, both for datasets with multiple tasks, and for the entire collection of datasets.