"""Task specific data processing functions for social dimensions."""

import json
import random
from dataclasses import asdict
from dataclasses import dataclass
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer
from trl.trainer import ConstantLengthDataset

from social_llama.config import DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED
from social_llama.data_processing.dataclass import DataClass
from social_llama.data_processing.dataset_configs import SOCIAL_DIMENSIONS_CONFIG
from social_llama.utils import save_json


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

    def preprocess(self, tokenizer) -> None:
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

        train_dataset = ConstantLengthDataset(
            tokenizer,
            train_data,
            formatting_func=self._prompt_function,
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

    def _preprocess_function(self, examples) -> dict:
        """Preprocess the data.

        Args:
            examples (dict): Dictionary of examples

        Returns:
            dict: Dictionary of examples
        """
        # Make a list of all labels that are not 0
        labels = []
        for label in self.config.labels:
            if examples[label] > 0:
                labels.append(label)
        examples["labels"] = labels

        #

        return examples

    def generate_prompt(self, example: dict) -> str:
        """Generate a prompt for the example.

        Args:
            example (dict): Dictionary containing the example

        Returns:
            str: Prompt for the example
        """
        return SOCIAL_DIMENSIONS_CONFIG.prompt_template.format(
            text=example["text"],
            social_dimensions=", ".join(example["labels"]),
        )


if __name__ == "__main__":
    social_dimensions = SocialDimensions()
    social_dimensions.get_data()

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    train_data, valid_data = social_dimensions.preprocess(tokenizer=tokenizer)
    a = 1
