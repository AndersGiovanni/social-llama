"""Task specific data processing functions for social dimensions."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import pandas as pd
from datasets import Dataset
from datasets import DatasetDict
from datasets import Features
from datasets import Value
from datasets import interleave_datasets
from datasets import load_dataset
from trl.trainer import ConstantLengthDataset
from typing_extensions import override

from social_llama.config import DATA_DIR_EVALUATION_SOCKET
from social_llama.data_processing.dataclass import DataClass
from social_llama.data_processing.dataset_configs import SOCIAL_DIMENSIONS_CONFIG


@dataclass
class Sample:
    """Dataclass for a sample in the Social Dimensions dataset."""

    text: str
    label: str
    task: str

    def __getitem__(self, idx: str) -> Any:
        """Get the item at the index."""
        return self.__dict__[idx]


@dataclass
class Socket(DataClass):
    """Dataclass for the Social Dimensions dataset.

    Atributes:
        data (DatasetDict, Dataset, None): Dataset or DatasetDict
        config (DatasetConfig): DatasetConfig object
    """

    def __init__(self, task: str, model: str) -> None:
        """Initialize the SocialDimensions class."""
        super().__init__(SOCIAL_DIMENSIONS_CONFIG, task=task, model=model)
        self.socket: pd.DataFrame = pd.read_csv(
            DATA_DIR_EVALUATION_SOCKET / "socket_prompts.csv"
        )
        # Select the ones which type is CLS
        self.socket = self.socket[self.socket["type"].isin(["CLS"])]
        self.data: Union[DatasetDict, DatasetDict, Dataset, None] = None
        self.labels: Dict[str, List[str]] = defaultdict(list)

    def __getitem__(self, index) -> Any:
        """Get the item at the index."""
        return super().__getitem__(index)

    @override
    def get_data(self) -> None:
        """Get the data."""
        tasks = self.socket["task"].values.tolist()

        features = Features(
            {
                "text": Value("string"),
                "label": Value("string"),
                "task": Value("string"),
            }
        )

        datasets_list: List[DatasetDict] = []

        # Get the data for each task
        for task in tasks:
            dataset: DatasetDict = load_dataset("Blablablab/SOCKET", task)

            # if length is more than 5000, randomly sample 5000
            if len(dataset["train"]) > 5000:
                dataset["train"] = dataset["train"].shuffle(seed=42).select(range(5000))

            # if length is more than 2000, randomly sample 2000
            if len(dataset["validation"]) > 2000:
                dataset["validation"] = (
                    dataset["validation"].shuffle(seed=42).select(range(2000))
                )

            # Remove the test set
            del dataset["test"]

            # For each sample in the dataset, add the task
            for split in dataset:
                dataset[split] = dataset[split].map(
                    lambda example: {
                        "text": example["text"],
                        "label": example["label"],
                        "task": task,
                    }
                )

            self.labels[task] = dataset["train"].features["label"].names
            dataset = dataset.cast(features)
            datasets_list.append(dataset)

        # Consider probabilities
        # probabilities = [1 / len(datasets_list)] * len(datasets_list)

        train_data = interleave_datasets(
            [dataset["train"] for dataset in datasets_list], seed=42
        )
        test_data = interleave_datasets(
            [dataset["validation"] for dataset in datasets_list], seed=42
        )

        self.train_data = train_data
        self.test_data = test_data

    @override
    def preprocess_sft(self) -> Tuple[ConstantLengthDataset, ConstantLengthDataset]:
        """Preprocess the data."""
        print(
            f"Size of the train set: {len(self.train_data)}. Size of the test set: {len(self.test_data)}"  # type: ignore
        )

        chars_per_token = self.chars_token_ratio(self.train_data, self.tokenizer)
        print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

        if self.task == "few-shot":
            raise NotImplementedError
            # Construct the dataset as few-shot
            # self.train_data = self._apply_few_shot_prompt_stf(self.train_data)
            # self.test_data = self._apply_few_shot_prompt_stf(self.test_data)

        formatting_func: Callable = self._prompt_function

        train_dataset = ConstantLengthDataset(
            self.tokenizer,
            self.train_data,
            formatting_func=formatting_func,
            infinite=True,
            seq_length=1024,
            chars_per_token=chars_per_token,
        )

        test_dataset = ConstantLengthDataset(
            self.tokenizer,
            self.test_data,
            formatting_func=formatting_func,
            infinite=True,
            seq_length=1024,
            chars_per_token=chars_per_token,
        )

        return train_dataset, test_dataset

    @override
    def _prompt_function(
        self, example: Union[Sample, List[Sample]], is_q_a: bool = False
    ) -> str:
        """Generate a prompt for the example.

        Args:
            example (Sample): Sample of a social dimensions example
            is_q_a (bool, optional): Whether the example is a question/answering example. Defaults to False.

        Returns:
            str: Prompt for the example

        Raises:
            ValueError: If the task is not supported.
        """
        chat: List[Dict[str, str]] = self.llama_config.get_chat_template()

        # If we want to make a custom prompt.
        chat[0]["content"] = chat[0]["content"].format(prompt_prefix="")

        prompt = self.socket[self.socket["task"] == example["task"]]["question"].iloc[0]

        if self.task == "zero-shot":
            if is_q_a:
                task_prompt = (
                    prompt.format(
                        text=example["text"],
                    )
                    + f" You can choose from the following labels: {', '.join(self.labels[example['task']])}\nAnswer:"
                )
            else:
                task_prompt = (
                    prompt.format(
                        text=example["text"],
                    )
                    + f" You can choose from the following labels: {', '.join(self.labels[example['task']])}\nAnswer: {example['label']}"
                )

        elif self.task == "few-shot":
            raise NotImplementedError
        elif self.task == "cot":
            raise NotImplementedError
        else:
            raise ValueError(f"Type {type} is not supported.")

        chat.append(
            {
                "role": "user",
                "content": task_prompt,
            }
        )

        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )


if __name__ == "__main__":
    socket = Socket(task="zero-shot", model="meta-llama/Llama-2-7b-chat-hf")

    socket.get_data()

    train, test = socket.preprocess_sft()

    a = 1
