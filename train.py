from hatecomp import NAACL
from transformers import AutoTokenizer, AutoModelForSequenceClassification

raw_dataset = NAACL()

for item in raw_dataset:
    print(item)
    break
