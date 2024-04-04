"""Task specific data processing functions for social dimensions."""

import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from trl.trainer import ConstantLengthDataset
from typing_extensions import override

from social_llama.data_processing.dataclass import DataClass
from social_llama.data_processing.dataset_configs import SOCIAL_DIMENSIONS_CONFIG
from social_llama.reverse_instructions.instruction_configs import (
    ReverseInstructionsPrompts,
)


warnings.filterwarnings("ignore")


@dataclass
class Sample:
    """Dataclass for a sample in the Social Dimensions dataset."""

    instruction: str
    text: str
    label: str
    task: str
    label_options: List[str]

    def __getitem__(self, idx: str) -> Any:
        """Get the item at the index.

        Args:
            idx (str): Index

        Raises:
            KeyError: If the index is not a valid attribute of the class.
        """
        try:
            return getattr(self, idx)
        except AttributeError as exc:
            raise KeyError(
                f"{idx} is not a valid attribute of {type(self).__name__}"
            ) from exc


@dataclass
class InstructionSocket(DataClass):
    """Dataclass for the Social Dimensions dataset.

    Atributes:
        data (DatasetDict, Dataset, None): Dataset or DatasetDict
        config (DatasetConfig): DatasetConfig object
    """

    def __init__(self, task: str, model: str) -> None:
        """Initialize the SocialDimensions class."""
        super().__init__(SOCIAL_DIMENSIONS_CONFIG, task=task, model=model)

        # Select the ones which type is CLS
        self.data: Union[DatasetDict, DatasetDict, Dataset, None]
        self.labels: Dict[str, List[str]] = defaultdict(list)
        self.reverse_instructions_prompts = ReverseInstructionsPrompts()

    def __getitem__(self, index) -> Any:
        """Get the item at the index."""
        return super().__getitem__(index)

    @override
    def get_data(self) -> None:
        """Get the data."""
        dataset: DatasetDict = load_dataset(
            "AndersGiovanni/instructions-SOCKET",
            num_proc=1,
            # download_mode="force_redownload",
        )  # type: ignore

        # Remove the test set
        del dataset["test"]

        # Shuffle the dataset
        dataset = dataset.shuffle(seed=42)

        # Get the labels
        for row in dataset["train"]:
            task = row["task"]
            label_options = row["label_options"]

            # Check if this task already has these label_options; if not, add them
            if task not in self.labels:
                self.labels[task] = label_options

        self.train_data = dataset["train"]
        self.test_data = dataset["validation"]

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
        chat: List[Dict[str, str]] = self.llama_config.get_chat_template(
            "system" if "llama" in self.model else "user"
        )

        # If we want to make a custom prompt.
        chat[0]["content"] = chat[0]["content"].format(prompt_prefix="")

        prompt: str = self.reverse_instructions_prompts.instruction_cls()

        if self.task == "zero-shot":
            if is_q_a:
                task_prompt = prompt.format(
                    instruction=example["instruction"],
                    text=example["text"],
                    label_list=", ".join(example["label_options"]),
                    label="",
                )
            else:
                task_prompt = prompt.format(
                    instruction=example["instruction"],
                    text=example["text"],
                    label_list=", ".join(example["label_options"]),
                    label=example["label"],
                )

        elif self.task == "few-shot":
            raise NotImplementedError
        elif self.task == "cot":
            raise NotImplementedError
        else:
            raise ValueError(f"Type {type} is not supported.")

        if "llama" in self.model:
            chat.append(
                {
                    "role": "user",
                    "content": task_prompt,
                }
            )
        else:
            chat[0] = {
                "role": "user",
                "content": f"{chat[0]['content']} {task_prompt}",  # Gemma is not trained with a system prompt
            }

        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )  # type: ignore

    @override
    def preprocess_dpo(self) -> Tuple[Dataset, Dataset]:
        """Preprocess for DPO. The data needs Q&A format."""
        original_columns = self.train_data.column_names

        self.train_data = self.train_data.map(
            self._convert_to_q_and_a,
            # batched=True,
            remove_columns=original_columns,
            drop_last_batch=True,
        )
        self.test_data = self.test_data.map(
            self._convert_to_q_and_a,
            # batched=True,
            remove_columns=original_columns,
            drop_last_batch=True,
        )

        return self.train_data, self.test_data

    @override
    def _convert_to_q_and_a(
        self, samples: Union[Sample, List[Sample], LazyRow]
    ) -> Dict[str, str]:
        """Convert the dataset to a question and answer dataset.

        Args:
            samples (Union[Sample, List[Sample]], LazyRow): Sample (if zero-shot) or list of samples (if few-shot or CoT)

        Returns:
            Dict: Dict with the prompt, chosen response, and rejected response
        """
        return {
            "prompt": self._prompt_function(samples, is_q_a=True),  # type: ignore
            "chosen": samples["label"],  # type: ignore
            "rejected": self.sample_rejected_label(self.labels[samples["task"]], samples["label"]),  # type: ignore
        }

    def sample_rejected_label(self, labels: List[str], exception: str):
        """Randomly sample a label from the list of labels, which is not equal to the chosen/correct label."""
        while True:
            choice = random.choice(labels)
            if choice != exception:
                return choice


if __name__ == "__main__":
    socket = InstructionSocket(task="zero-shot", model="meta-llama/Llama-2-7b-chat-hf")

    socket.get_data()

    # train, test = socket.preprocess_sft()

    train, test = socket.preprocess_dpo()

    a = 1
