"""Combine datasets."""

from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from datasets import Dataset
from datasets import DatasetDict
from datasets import interleave_datasets
from transformers import AutoTokenizer
from trl.trainer import ConstantLengthDataset

from social_llama.config import LlamaConfigs
from social_llama.data_processing.social_dimensions import (
    Sample as SocialDimensionsSample,
)
from social_llama.data_processing.social_dimensions import SocialDimensions
from social_llama.data_processing.socket import Sample as SocketSample
from social_llama.data_processing.socket import Socket


class Combined:
    """Combine datasets."""

    def __init__(self, model: str) -> None:
        """Initialize the Combined class."""
        self.model = model
        self.social_dimensions_dataset = SocialDimensions(task="zero-shot", model=model)
        self.socket_dataset = Socket(task="zero-shot", model=model)
        self.train_data: Union[DatasetDict, Dataset, None] = None
        self.test_data: Union[DatasetDict, Dataset, None] = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=True,
            add_special_tokens=False,
            add_eos_token=False,
            add_bos_token=False,
        )
        self.tokenizer.use_default_system_prompt = False
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = (
            "right"  # Fix weird overflow issue with fp16 training
        )
        self.llama_config = LlamaConfigs()

    def get_data(self) -> None:
        """Get the data from the datasets."""
        self.social_dimensions_dataset.get_data()
        self.socket_dataset.get_data()

        # Interleave the datasets
        self.train_data = interleave_datasets(
            [
                self.social_dimensions_dataset.train_data,
                self.socket_dataset.train_data,
            ],
            seed=42,
            stopping_strategy="first_exhausted",
        )

        self.test_data = interleave_datasets(
            [
                self.social_dimensions_dataset.test_data,
                self.socket_dataset.test_data,
            ],
            seed=42,
            stopping_strategy="first_exhausted",
        )

    def preprocess_sft(self) -> Tuple[ConstantLengthDataset, ConstantLengthDataset]:
        """Preprocess the data."""
        chars_per_token_social_dimensions = (
            self.social_dimensions_dataset.chars_token_ratio(
                self.social_dimensions_dataset.train_data, self.tokenizer
            )
        )
        chars_per_token_socket = self.socket_dataset.chars_token_ratio(
            self.socket_dataset.train_data, self.tokenizer
        )
        print(
            f"The character to token ratio of the social dimensions dataset is: {chars_per_token_social_dimensions:.2f}"
        )
        print(
            f"The character to token ratio of the socket dataset is: {chars_per_token_socket:.2f}"
        )

        avg_chars_per_token = (
            chars_per_token_social_dimensions + chars_per_token_socket
        ) / 2

        formatting_func: Callable = self._prompt_function

        train_dataset = ConstantLengthDataset(
            self.tokenizer,
            self.train_data,
            formatting_func=formatting_func,
            infinite=True,
            seq_length=1024,
            chars_per_token=avg_chars_per_token,
        )

        test_dataset = ConstantLengthDataset(
            self.tokenizer,
            self.test_data,
            formatting_func=formatting_func,
            infinite=True,
            seq_length=1024,
            chars_per_token=avg_chars_per_token,
        )

        return train_dataset, test_dataset

    def _prompt_function(
        self,
        example: Union[
            SocialDimensionsSample,
            List[SocialDimensionsSample],
            SocketSample,
            List[SocketSample],
        ],
        is_q_a: bool = False,
    ) -> str:
        """Generate a prompt for the example.

        Args:
            example (Sample): Sample of a social dimensions example
            is_q_a (bool, optional): Whether the example is a question/answering example. Defaults to False.

        Returns:
            str: Prompt for the example
        """
        chat: List[Dict[str, str]] = self.llama_config.get_chat_template()

        # If we want to make a custom prompt.
        chat[0]["content"] = chat[0]["content"].format(prompt_prefix="")

        if example["task"] == "social-dimensions":
            task_prompt = self.social_dimensions_dataset.config.prompt_template.format(
                text=example["text"],
                response_good=example["response_good"] if not is_q_a else "",
            )
        else:
            prompt = self.socket_dataset.socket[
                self.socket_dataset.socket["task"] == example["task"]
            ]["question"].iloc[0]

            if is_q_a:
                task_prompt = (
                    prompt.format(
                        text=example["text"],
                    )
                    + f" You can choose from the following labels: {', '.join(self.socket_dataset.labels[example['task']])}\nAnswer:"
                )
            else:
                task_prompt = (
                    prompt.format(
                        text=example["text"],
                    )
                    + f" You can choose from the following labels: {', '.join(self.socket_dataset.labels[example['task']])}\nAnswer: {self.socket_dataset.labels[example['task']][example['label']]}"
                )

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
    combined = Combined(model="meta-llama/Llama-2-7b-hf")
    combined.get_data()
    train_dataset, test_dataset = combined.preprocess_sft()
