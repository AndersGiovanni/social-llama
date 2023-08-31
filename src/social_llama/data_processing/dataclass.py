"""Dataclass to abstract the some data processing."""

from typing import Union

from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from social_llama.config import DatasetConfig


class DataClass(TorchDataset):
    """Dataclass abstraction for the datasets. This gives us a unified framework."""

    def __init__(self, config) -> None:
        """Initialize the DataClass."""
        super().__init__()
        self.data: Union[
            DatasetDict, Dataset, IterableDataset, IterableDatasetDict, None
        ] = None
        self.config: DatasetConfig = config

    def set_data(
        self, data: Union[DatasetDict, Dataset, IterableDataset, IterableDatasetDict]
    ) -> None:
        """Sets the data.

        Args:
            data (Union[DatasetDict, Dataset]): Dataset or DatasetDict
        """
        self.data = data

    def get_data(self) -> None:
        """Reads the data from the data directory.

        Specific for individual datasets, so this function should be overwritten by the child class.
        """
        raise NotImplementedError

    def preprocess(self, tokenizer) -> None:
        """This function should be overwritten by the child class.

        It should preprocess the data for the model.
        This includes tokenization, label preprocessing, and column formatting
        """
        raise NotImplementedError

    def _prompt_function(self, example: str) -> str:
        """Prompt function for the dataset.

        Args:
            example (str): Example from the dataset

        Returns:
            str: Prompt for the example
        """
        return self.config.prompt_template.format(
            text=example["text"], response_good=example["response_good"]
        )

    def chars_token_ratio(self, dataset, tokenizer, nb_examples=400):
        """Estimate the average number of characters per token in the dataset."""
        total_characters, total_tokens = 0, 0
        for _, example in tqdm(
            zip(range(nb_examples), iter(dataset), strict=True), total=nb_examples
        ):
            text = self._prompt_function(example)
            total_characters += len(text)
            if tokenizer.is_fast:
                total_tokens += len(tokenizer(text).tokens())
            else:
                total_tokens += len(tokenizer.tokenize(text))

        return total_characters / total_tokens
