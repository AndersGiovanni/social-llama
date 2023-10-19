"""Dataclass to abstract the some data processing."""

from abc import abstractmethod
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from social_llama.config import DatasetConfig
from social_llama.config import LlamaConfigs


class DataClass(TorchDataset):
    """Dataclass abstraction for the datasets. This gives us a unified framework."""

    def __init__(self, config: DatasetConfig, task: str, model: str) -> None:
        """Initialize the DataClass."""
        super().__init__()
        self.data: Union[
            DatasetDict, Dataset, IterableDataset, IterableDatasetDict, None
        ] = None
        self.config: DatasetConfig = config
        self.task: str = task
        self.llama_config = LlamaConfigs()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.tokenizer.use_default_system_prompt = False

    def set_data(
        self, data: Union[DatasetDict, Dataset, IterableDataset, IterableDatasetDict]
    ) -> None:
        """Sets the data.

        Args:
            data (Union[DatasetDict, Dataset]): Dataset or DatasetDict
        """
        self.data = data

    @abstractmethod
    def get_data(self) -> None:
        """Reads the data from the data directory.

        Specific for individual datasets, so this function should be overwritten by the child class.
        """

    @abstractmethod
    def preprocess_sft(self) -> Any:
        """This function should be overwritten by the child class.

        It should preprocess the data for the model.
        This includes tokenization, label preprocessing, and column formatting
        """

    @abstractmethod
    def preprocess_dpo(self) -> Any:
        """This function should be overwritten by the child class.

        It should preprocess the data for the model.
        This includes tokenization, label preprocessing, and column formatting
        """

    @abstractmethod
    def _extract_few_shot_examples(
        self, dataset: dict, seed: int = 42
    ) -> Tuple[List[Any], Dataset]:
        """Extracts the few shot examples from the dataset.

        Function should be overwritten by the child class.
        """

    @abstractmethod
    def _apply_few_shot_prompt_stf(self, dataset, seed) -> None:
        """Applies the few shot prompt to the dataset for SFT.

        Function should be overwritten by the child class.
        """

    @abstractmethod
    def _apply_few_shot_prompt_dpo(self, dataset, seed) -> None:
        """Applies the few shot prompt to the dataset for DPO.

        Function should be overwritten by the child class.
        """

    @abstractmethod
    def _prompt_function(self, example: Any) -> str:
        """Prompt function for the dataset.

        Args:
            example (str): Example from the dataset

        Returns:
            str: Prompt for the example
        """

    def chars_token_ratio(self, dataset, tokenizer, nb_examples=400):
        """Estimate the average number of characters per token in the dataset."""
        total_characters, total_tokens = 0, 0
        for _, example in tqdm(
            zip(range(nb_examples), iter(dataset)), total=nb_examples
        ):
            text = self._prompt_function(example)
            # text = example["text"]
            total_characters += len(text)
            if tokenizer.is_fast:
                total_tokens += len(tokenizer(text).tokens())
            else:
                total_tokens += len(tokenizer.tokenize(text))

        return total_characters / total_tokens

    @abstractmethod
    def _convert_to_q_and_a(self, samples: List[Any]) -> Dataset:
        """Load dataset into the question answering format."""
