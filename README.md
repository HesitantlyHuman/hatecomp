# Hate Datasets Compilation
Contains 17 different hate, violence, and discrimination speech datasets, along with anotations on where they were found, the data format and method for collection and labeling. Each dataset is kept in the original file structure and placed inside the respective folder's `data` file, in case a more recent version is obtained. (Links for the source are in each ABOUT.md file)

## Using
[TODO]: Detail how to download and use

## Datasets
Additional information about each dataset can be found the corresponding ABOUT.md file. 

### Working
Currently the following datasets are implemented:
- [ZeerakTalat NAACL]()
- ZeerakTalat NLPCSS
- HASOC
- Vicomtech

### Notes

Of those the `MLMA Dataset` and `Online Intervention Dataset` only contain hateful posts, instead labelling other features such as the target of hate.

## Training
`hatecomp` provides some basic training tools to integrate into the [huggingface](https://github.com/huggingface) :hugs: Trainer API. A full example of how to train a model using the hatecomp datasets can be found in `train.py`

### Results
Here is a list of results acheived on various datasets with Huggingface models, along with the SOTA performance (as best I could find). Since it is not always possible to find SOTA scores for obscure datsets measured with a particular metric, the hatecomp score is selected to match whatever SOTA could be found. (The links are locations where the SOTA reference was found. If you are looking for citations, please refer to teh `About.md` for each dataset)

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
Collect and organize the Cornell Conversational Analysis Toolkit datasets found [here](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit#datasets)
Add sentiment lexicon https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.html
Update package and requirements.txt