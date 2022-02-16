from typing import Optional, Union, Any, Mapping
from hatecomp.base import DataLoader as HatecompDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
import torch
from transformers import Trainer


class HatecompTrainer(Trainer):
    def get_train_dataloader(self) -> TorchDataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()

        return HatecompDataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(
        self, eval_dataset: Optional[Dataset] = None
    ) -> TorchDataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return HatecompDataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> TorchDataLoader:
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return HatecompDataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _prepare_input(
        self, data: Union[torch.Tensor, Any]
    ) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        Modified from the hugginface trainer _prepare_input to not consider the first item of a tuple, or the id
        item of a dictionary. Tensors cannot have string values, so they are ignored.
        """
        if isinstance(data, Mapping):
            return type(data)(
                {k: self._prepare_input(v) for k, v in data.items() if not k == "id"}
            )
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data[1:])
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self.args.device)
            if self.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data
