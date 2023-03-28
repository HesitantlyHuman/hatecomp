from typing import Callable, List
import torch
from transformers import AutoTokenizer
import logging

# Suppress the too long message for tokinization, since we will handle that ourselves
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class HatecompTokenizer:
    def __init__(
        self,
        tokenizer: Callable[[List[str]], List[int]],
        max_length: int = 512,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self._start_token_length = int(max_length / 2)
        self._end_token_length = max_length - self._start_token_length

    def __call__(self, tokenization_input: str):
        tokens = self.tokenizer(
            tokenization_input, padding=True, truncation=False, return_tensors="pt"
        )

        # If the length of the tokens is greater than the max_length, then
        # grab the first half of the tokens and the last half of the tokens
        # and concatenate them together.
        if tokens["input_ids"].shape[1] > self.max_length:
            for key in tokens:
                tokens[key] = torch.cat(
                    (
                        tokens[key][:, : self._start_token_length],
                        tokens[key][:, -self._end_token_length :],
                    ),
                    dim=1,
                )

        return tokens

    @classmethod
    def from_huggingface_pretrained(
        self, model_name: str, max_length: int = 512
    ) -> "HatecompTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return HatecompTokenizer(tokenizer, max_length=max_length)


if __name__ == "__main__":
    tokenizer = HatecompTokenizer("roberta-base")
    print(tokenizer("This is a test of the tokenizer.")["input_ids"].shape)
    # now let's test a string with more than 512 tokens
    print(
        tokenizer(
            """Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.dense.weight', 'lm_head.decoder.weight', 'roberta.pooler.dense.weight', 'lm_head.layer_norm.weight', 'lm_head.dense.bias', 'roberta.pooler.dense.bias', 'lm_head.bias', 'lm_head.layer_norm.bias']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.out_proj.bias', 'classifier.out_proj.weight', 'classifier.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.dense.weight', 'lm_head.decoder.weight', 'roberta.pooler.dense.weight', 'lm_head.layer_norm.weight', 'lm_head.dense.bias', 'roberta.pooler.dense.bias', 'lm_head.bias', 'lm_head.layer_norm.bias']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.out_proj.bias', 'classifier.out_proj.weight', 'classifier.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
            """
        )["input_ids"].shape
    )
