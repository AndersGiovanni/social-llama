"""Task specific data processing functions for social dimensions."""

import itertools
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Union

from datasets import Dataset
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from trl.trainer import ConstantLengthDataset
from typing_extensions import override

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

    def __getitem__(self, idx: str) -> Any:
        """Get the item at the index."""
        return self.__dict__[idx]


@dataclass
class SocialDimensions(DataClass):
    """Dataclass for the Social Dimensions dataset.

    Atributes:
        data (DatasetDict, Dataset, None): Dataset or DatasetDict
        config (DatasetConfig): DatasetConfig object
    """

    def __init__(self, task: str, model: str) -> None:
        """Initialize the SocialDimensions class."""
        super().__init__(SOCIAL_DIMENSIONS_CONFIG, task=task, model=model)

    def __getitem__(self, index) -> Any:
        """Get the item at the index."""
        return super().__getitem__(index)

    def simple_json_solution(self) -> None:
        """Reads the data from the data directory.

        This converts the data from the raw format to a format that aligns with the tutorial from TRL.
        """
        with open(self.config.path) as f:
            data = json.load(f)

        processes_data: List[dict[str, Any]] = []

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
                    idx=str(idx),
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
        train_data = load_dataset(
            "json",
            data_files=str(self.config.path / "train.json"),
            split="train",
        )

        test_data = load_dataset(
            "json",
            data_files=str(self.config.path / "test.json"),
            split="train",
        )

        self.set_data(train_data=train_data, test_data=test_data)

    @override
    def preprocess_sft(self) -> Tuple[ConstantLengthDataset, ConstantLengthDataset]:
        """Preprocess the data."""
        print(
            f"Size of the train set: {len(self.train_data)}. Size of the test set: {len(self.test_data)}"  # type: ignore
        )

        chars_per_token = self.chars_token_ratio(self.train_data, self.tokenizer)
        print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

        if self.task == "few-shot":
            # Construct the dataset as few-shot
            self.train_data = self._apply_few_shot_prompt_stf(self.train_data)
            self.test_data = self._apply_few_shot_prompt_stf(self.test_data)

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

    def _prompt_function(self, example: Sample, is_q_a: bool = False) -> str:
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

        chat[0]["content"] = chat[0]["content"].format(
            prompt_prefix=self.config.prompt_prefix
        )

        if self.task == "zero-shot":
            task_prompt = self.config.prompt_template.format(
                text=example["text"],
                response_good=example["response_good"] if not is_q_a else "",
            )
        elif self.task == "few-shot":  # TODO: Fix this
            task_prompt = example["text"]
        elif self.task == "cot":
            task_prompt = self.config.prompt_template_cot.format(
                text=example["text"],
                response_good=example["response_good"] if not is_q_a else "",
                dimension_description=self.config.cot_info_dict[  # type: ignore
                    example["response_good"]
                ],
                h_text=example["h_text"],
            )
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

    def _extract_few_shot_examples(
        self, dataset, seed: int = 42
    ) -> Tuple[List[Sample], Dataset]:
        """Extracts the few shot examples from the dataset."""
        # Define variables
        num_few_shot_examples: int = self.config.num_few_shot_examples
        labels: Iterator = itertools.cycle(self.config.labels)
        few_shot_examples: List[Sample] = []

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

    def _extract_few_shot_examples_v2(
        self, dataset, num_few_shot_examples: int = 3, seed: int = 42
    ) -> Tuple[List[dict], List[dict]]:
        # Prepare a dictionary to hold examples for each label
        label_examples = defaultdict(list)

        # Populate the dictionary
        for example in dataset:
            if example["response_good"] in self.config.labels:
                label_examples[example["response_good"]].append(example)

        # Prepare the few_shot_examples list
        few_shot_examples = []
        labels_cycle = itertools.cycle(self.config.labels)

        while len(few_shot_examples) < num_few_shot_examples:
            label = next(labels_cycle)

            if label_examples[label]:
                example = label_examples[label].pop(0)
                few_shot_examples.append(example)

        random.seed(seed)
        random.shuffle(few_shot_examples)

        # Remove used examples from the original dataset
        used_texts = {example["text"] for example in few_shot_examples}
        remaining_dataset = [x for x in dataset if x["text"] not in used_texts]

        return few_shot_examples, remaining_dataset

    def _make_few_shot_example(self, few_shot_examples: List[Sample]) -> str:
        """Make a few shot example."""
        samples_with_prompt = [
            self.config.prompt_template.format(
                text=example["text"], response_good=example["response_good"]
            )
            for example in few_shot_examples
        ]
        return "\n".join(samples_with_prompt)

    @override
    def _apply_few_shot_prompt_stf(self, dataset, seed: int = 42) -> Dataset:
        """Applies the few shot prompt to the dataset."""
        # Shuffle dataset
        shuffled_data = dataset.shuffle(seed=seed)

        # All the samples after making them into a few shot example
        few_shot_collection = []
        while shuffled_data.num_rows >= self.config.num_few_shot_examples:
            # Extract few shot examples
            few_shot_examples, shuffled_data = self._extract_few_shot_examples_v2(
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

    @override
    def _convert_to_q_and_a(
        self, samples: Union[Sample, List[Sample], LazyRow]
    ) -> Dict:
        """Convert the dataset to a question and answer dataset.

        Args:
            samples (Union[Sample, List[Sample]], LazyRow): Sample (if zero-shot) or list of samples (if few-shot or CoT)

        Returns:
            Dict: Dict with the prompt, chosen response, and rejected response

        Raises:
            ValueError: If the task is not supported
        """
        if self.task == "zero-shot":
            return {
                "prompt": self._prompt_function(samples, is_q_a=True),  # type: ignore
                "chosen": samples["response_good"],  # type: ignore
                "rejected": samples["response_bad"],  # type: ignore
            }
        elif self.task == "few-shot":
            return {
                "prompt": "".join(
                    [
                        f"Text: {text}\n\nAnswer: {response_good}\n"
                        for text, response_good in zip(
                            samples["text"][:-1], samples["response_good"][:-1]  # type: ignore
                        )
                    ]
                )
                + f"Text: {samples['text'][-1]}\n\nAnswer: ",  # type: ignore
                "chosen": samples["response_good"][-1],  # type: ignore
                "rejected": samples["response_bad"][-1],  # type: ignore
            }
        elif self.task == "cot":
            return {
                "prompt": "".join(
                    [
                        f"Text: {text}\nThe text exhibits {self.config.cot_info_dict[response_good]}. In particular in the part '{h_text!r}'.\nAnswer: {response_good}\n"  # type: ignore
                        for text, h_text, response_good in zip(
                            samples["text"][:-1],  # type: ignore
                            samples["h_text"][:-1],  # type: ignore
                            samples["response_good"][:-1],  # type: ignore
                        )
                    ]
                )
                + f"Text: {samples['text'][-1]}\n",  # type: ignore
                "chosen": f"The text exhibits {self.config.cot_info_dict[samples['response_good'][-1]]}. In particular in the part '{samples['h_text'][-1]!r}'.\nAnswer: {samples['response_good'][-1]}\n",  # type: ignore
                "rejected": f"The text exhibits {self.config.cot_info_dict[samples['response_bad'][-1]]}. In particular in the part '{samples['h_text'][-1]!r}'.\nAnswer: {samples['response_bad'][-1]}\n",  # type: ignore
            }
        else:
            raise ValueError(f"Type {type} is not supported.")

    def _apply_few_shot_prompt_dpo(self, dataset, seed: int = 42) -> Dataset:
        """Applies the few shot prompt to the dataset."""
        # Shuffle dataset
        shuffled_data = dataset.shuffle(seed=seed)

        # All the samples after making them into a few shot example
        few_shot_collection = []

        while len(shuffled_data) >= self.config.num_few_shot_examples:
            # Extract few shot examples
            few_shot_examples, shuffled_data = self._extract_few_shot_examples_v2(
                shuffled_data, seed=seed
            )

            # Add few shot examples to few shot dataset
            few_shot_collection.append(
                {
                    "idx": [example["idx"] for example in few_shot_examples],
                    "text": [example["text"] for example in few_shot_examples],
                    "h_text": [example["h_text"] for example in few_shot_examples],
                    "response_good": [
                        example["response_good"] for example in few_shot_examples
                    ],
                    "response_bad": [
                        example["response_bad"] for example in few_shot_examples
                    ],
                }
            )

            # Every 1000 examples print the number of examples left
            if len(few_shot_collection) % 1000 == 0:
                print(
                    f"Number of examples left: {len(shuffled_data)}. Number of few shot examples: {len(few_shot_collection)}"
                )

        # Define new empty DatasetDict
        few_shot_dataset = Dataset.from_list(few_shot_collection)

        return few_shot_dataset

    @override
    def preprocess_dpo(self) -> Tuple[Dataset, Dataset]:
        """Preprocess for DPO. The data needs Q&A format."""
        original_columns = self.train_data.column_names

        if self.task == "few-shot" or self.task == "cot":
            self.train_data = self._apply_few_shot_prompt_dpo(self.train_data)
            self.test_data = self._apply_few_shot_prompt_dpo(self.test_data)

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


if __name__ == "__main__":
    social_dimensions = SocialDimensions(task="cot", model="meta-llama/Llama-2-70b-hf")
    social_dimensions.get_data()

    train_data_, valid_data_ = social_dimensions.preprocess_dpo()
