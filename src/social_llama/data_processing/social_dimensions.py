"""Task specific data processing functions for social dimensions."""

import itertools
import json
import os
import random
from dataclasses import asdict
from dataclasses import dataclass
from typing import Dict
from typing import Iterator
from typing import List

from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.trainer import ConstantLengthDataset

from social_llama.config import DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED
from social_llama.data_processing.dataclass import DataClass
from social_llama.data_processing.dataset_configs import SOCIAL_DIMENSIONS_CONFIG
from social_llama.utils import save_json


os.environ["TQDM_DISABLE"] = "1"  # This disables the tqdm progress bar


@dataclass
class Sample:
    """Dataclass for a sample in the Social Dimensions dataset."""

    idx: str
    text: str
    h_text: str
    response_good: str
    response_bad: str


@dataclass
class SocialDimensions(DataClass):
    """Dataclass for the Social Dimensions dataset.

    Atributes:
        data (DatasetDict, Dataset, None): Dataset or DatasetDict
        config (DatasetConfig): DatasetConfig object
    """

    def __init__(self) -> None:
        """Initialize the SocialDimensions class."""
        super().__init__(SOCIAL_DIMENSIONS_CONFIG)

    def simple_json_solution(self) -> None:
        """Reads the data from the data directory.

        This converts the data from the raw format to a format that aligns with the tutorial from TRL.
        """
        with open(self.config.path) as f:
            data = json.load(f)

        processes_data: List[Sample] = []

        for idx, example in enumerate(data):
            # Get labels and their values
            labels_counts = [
                (label, value)
                for label, value in example.items()
                if label in self.config.labels
            ]
            # All the labels that are not 0
            positive_labels = [label for label, value in labels_counts if value > 0]
            # All the labels that are 0
            negative_labels = [label for label, value in labels_counts if value == 0]

            # If there are no positive labels, skip the example
            if len(positive_labels) == 0:
                continue

            for positive in positive_labels:
                sample = Sample(
                    idx=idx,
                    text=example["text"],
                    h_text=example["h_text"],
                    response_good=positive,
                    # Randomly select a negative label
                    response_bad=random.choice(negative_labels),
                )
                processes_data.append(asdict(sample))
        save_json(
            DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED / "labeled_dataset_small.json",
            processes_data,
        )

    def get_data(self) -> None:
        """Reads the data from the data directory."""
        data = load_dataset(
            "json",
            data_files=str(self.config.path),
            split="train",
        )

        self.set_data(data)

    def preprocess(self, tokenizer, data_processing: str) -> None:
        """Preprocess the data."""
        # Train test split - WE NEED TO MAKE CORRECT SPLITS TO AVIOD CONTAMINATION!
        self.data = self.data.train_test_split(
            test_size=0.2,
            shuffle=True,
            seed=42,
        )

        train_data = self.data["train"]
        valid_data = self.data["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )

        chars_per_token = self.chars_token_ratio(train_data, tokenizer)
        print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

        if data_processing == "zero-shot":
            formatting_func = self._prompt_function
        elif data_processing == "few-shot":
            formatting_func = self._prompt_function_few_shot

            # Construct the dataset as few-shot
            train_data = self._apply_few_shot_prompt(train_data)
            valid_data = self._apply_few_shot_prompt(valid_data)
        elif data_processing == "cot":
            formatting_func = self._prompt_function_cot
        else:
            raise ValueError(
                f"Data processing method {data_processing} is not supported."
            )

        train_dataset = ConstantLengthDataset(
            tokenizer,
            train_data,
            formatting_func=formatting_func,
            infinite=True,
            seq_length=1024,
            chars_per_token=chars_per_token,
        )

        valid_dataset = ConstantLengthDataset(
            tokenizer,
            valid_data,
            formatting_func=self._prompt_function,
            infinite=True,
            seq_length=1024,
            chars_per_token=chars_per_token,
        )

        return train_dataset, valid_dataset

    def _prompt_function(self, example: Dict) -> str:
        """Generate a prompt for the example.

        Args:
            example (dict): Dictionary containing the example

        Returns:
            str: Prompt for the example
        """
        return self.config.prompt_template.format(
            text=example["text"], response_good=example["response_good"]
        )

    def _prompt_function_few_shot(self, example: Dict) -> str:
        """Generate a prompt for the example.

        Args:
            example (dict): Dictionary containing the example

        Returns:
            str: Prompt for the example
        """
        return example["text"]

    def _prompt_function_cot(self, example: Dict) -> str:
        """Generate a prompt for the example.

        Args:
            example (dict): Dictionary containing the example

        Returns:
            str: Prompt for the example
        """
        return self.config.prompt_template_cot.format(
            text=example["text"],
            response_good=example["response_good"],
            dimension_description=self.config.cot_info_dict[example["response_good"]],
            h_text=example["h_text"],
        )

    def _extract_few_shot_examples(self, dataset, seed: int = 42) -> List[Dict]:
        """Extracts the few shot examples from the dataset."""
        # Define variables
        num_few_shot_examples: int = self.config.num_few_shot_examples
        labels: Iterator = itertools.cycle(self.config.labels)
        few_shot_examples: List[Dict] = []

        # Extract few shot examples
        while len(few_shot_examples) < num_few_shot_examples:
            # Get next/another label
            label = next(labels)
            # Get example with that label
            example = dataset.filter(lambda x: x["response_good"] == label).select(
                range(1)
            )
            # Skip if example is None
            if len(example) == 0:
                continue
            example = example[0]
            # Add example to few shot examples
            few_shot_examples.append(example)
            # Remove example from dataset
            dataset = dataset.filter(lambda x: x["text"] != example["text"])

        random.seed(seed)
        random.shuffle(few_shot_examples)

        return few_shot_examples, dataset

    def _make_few_shot_example(self, few_shot_examples: Dict) -> str:
        """Make a few shot example."""
        samples_with_prompt = [
            self._prompt_function(example) for example in few_shot_examples
        ]
        return "\n".join(samples_with_prompt)

    def _apply_few_shot_prompt(self, dataset, seed: int = 42) -> None:
        """Applies the few shot prompt to the dataset."""
        # Shuffle dataset
        shuffled_data = dataset.shuffle(seed=seed)

        # All the samples after making them into a few shot example
        few_shot_collection = []
        while shuffled_data.num_rows >= self.config.num_few_shot_examples:
            # Extract few shot examples
            few_shot_examples, shuffled_data = self._extract_few_shot_examples(
                shuffled_data, seed=seed
            )

            # Add few shot examples to few shot dataset
            few_shot_collection.append(self._make_few_shot_example(few_shot_examples))

            # Every 1000 examples print the number of examples left
            if len(few_shot_collection) % 1000 == 0:
                print(
                    f"Number of examples left: {shuffled_data.num_rows}. Number of few shot examples: {len(few_shot_collection)}"
                )

        # Define new empty DatasetDict
        few_shot_dataset = Dataset.from_dict({"text": few_shot_collection})

        return few_shot_dataset


if __name__ == "__main__":
    social_dimensions = SocialDimensions()
    social_dimensions.get_data()
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_data, valid_data = social_dimensions.preprocess(
        tokenizer=tokenizer, data_processing="cot"
    )
    train_data.dataset = train_data.dataset.select(range(100))
    a = 1
