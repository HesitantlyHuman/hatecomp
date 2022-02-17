# Hate Datasets Compilation
Contains 17 different hate, violence, and discrimination speech datasets, along with anotations on where they were found, the data format and method for collection and labeling. Each dataset is kept in the original file structure and placed inside the respective folder's `data` file, in case a more recent version is obtained. (Links for the source are in each ABOUT.md file)

# Using
[TODO]: Detail how to download and use

# Datasets
Additional information about each dataset can be found the corresponding ABOUT.md file. 

## Working
Currently only the following datasets are implemented:
- ZeerakW
- HASOC
- Vicomtech

## Useful Dataset
The datasets collected which contain annotated or labeled english text are the following:
- Iron March Dataset
- Sexism Dataset
- UB Web Datasets
    - Facebook Comments
    - League of Legends
    - World of Warcraft
    - Twitter Harassment
- Wikipedia Talk
- Online Intervention Dataset
- MLMA Dataset
- HASOC English Dataset
- ZeerakW Twitter Dataset
- Vicomtech Hate Speech Dataset

Of those, the `"UB Web Datasets/Twitter Harassment"`, `ZeerakW Twitter Dataset` and `Sexism Dataset` require downloading Tweet content from Twitter.

Additionally, the `MLMA Dataset` and `Online Intervention Dataset` only contain hateful posts, instead labelling other features such as the target of hate.

# Training
`hatecomp` provides some basic training tools to integrate into the [huggingface]() :hugs: Trainer API. A full example of how to train a model using the hatecomp datasets can be found in `train.py`

## Results
Here is a list of results acheived on various datasets with Huggingface models, along with the SOTA performance (as best I could find). Since it is not always possible to find SOTA scores for obscure datsets measured with a particular metric, the hatecomp score is selected to match whatever SOTA could be found. (The links are locations where the SOTA reference was found. If you are looking for citations, please refer to teh `About.md` for each dataset)

| Dataset | Metric | SOTA | hatecomp/huggingface |
| -- | -- | -- | -- |
| [Vicomtech](https://arxiv.org/pdf/1809.04444.pdf) | Accuracy | 0.79 | **0.93** |
| [ZeerakTalat-NAACL](https://aclanthology.org/N16-2013.pdf) | F1 | 0.74 | NA |
| [ZeerakTalat-NLPCSS](https://aclanthology.org/W16-5618.pdf) | F1 | [0.86, 0.92] | NA |
| [HASOC](https://arxiv.org/pdf/2108.05927.pdf) | F1 (Macro) | 0.53 | NA |
| [TwitterSexism](https://aclanthology.org/W17-2902.pdf) | F1 (Macro) | 0.87 | NA |