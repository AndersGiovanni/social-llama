"""Evaluation of the model."""

import os
import time
from typing import Dict
from typing import List
from typing import Union

import pandas as pd
from datasets import Dataset
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import pipeline

from social_llama.config import DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS
from social_llama.config import DATA_DIR_EVALUATION_SOCKET
from social_llama.config import LlamaConfigs
from social_llama.data_processing.social_dimensions import SocialDimensions
from social_llama.evaluation.helper_functions import label_check
from social_llama.evaluation.helper_functions import label_finder
from social_llama.utils import get_device
from social_llama.utils import save_json


load_dotenv()


class Evaluator:
    """Evaluator for our tasks dataset."""

    def __init__(self, model_id: str) -> None:
        """Initialize the evaluator."""
        self.socket_tasks: List[str] = ["CLS", "REG", "PAIR", "SPAN"]
        self.model_id = model_id
        self.social_dimensions = SocialDimensions(
            task="zero-shot", model="meta-llama/Llama-2-7b-chat-hf"
        )
        self.social_dimensions.get_data()
        self.llama_config = LlamaConfigs()
        self.socket_prompts: pd.DataFrame = pd.read_csv(
            DATA_DIR_EVALUATION_SOCKET / "socket_prompts.csv"
        )
        self.generation_kwargs = {
            "max_new_tokens": 50,
            "temperature": 0.9,
            "truncate": 4096,
            # "stop_sequences": self.social_dimensions.config.labels,
        }
        self.generation_kwargs_local = {
            "max_new_tokens": 20,
            "temperature": 0.9,
        }
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        if model_id in ["meta-llama/Llama-2-7b-chat-hf"]:
            self.inference_client = InferenceClient(
                model=model_id, token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
            )
            self.use_inference_client = True
        else:
            self.config = AutoConfig.from_pretrained(model_id)
            self.llama_config = LlamaConfigs
            self.device = get_device()
            self.llm = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=self.tokenizer,
                # device=self.device,
                device_map="auto",
            )
            self.use_inference_client = False

    def predict(self, task: str = "social-dimensions") -> None:
        """Predict the labels for the test data."""
        if task == "social-dimensions":
            task_data = self._prepare_social_dim_test_data()
            predictions = []

            for sample in tqdm(task_data):
                prediction = self._predict(sample)

                prediction_processed = label_check(
                    prediction=prediction,
                    labels=self.social_dimensions.config.labels,
                )
                predictions.append(
                    {
                        "idx": sample["idx"],
                        "prompt": sample["prompt"],
                        "prediction": prediction,
                        "prediction_processed": prediction_processed,
                        "label": sample["label"],
                    }
                )
            save_json(
                DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS
                / f"{self.model_id}_predictions.json",
                predictions,
            )
        elif task == "socket":
            cls_tasks = self.socket_prompts[self.socket_prompts["type"] == "CLS"][
                "task"
            ]
            for task in cls_tasks:
                task_data, labels = self._prepare_socket_test_data(task=task)
                predictions = []

                for sample in tqdm(task_data):
                    prediction = self._predict(sample)

                    prediction_processed = label_check(
                        prediction=prediction,
                        labels=labels,
                    )
                    prediction_finder = label_finder(
                        prediction=prediction,
                        labels=labels,
                    )
                    predictions.append(
                        {
                            "idx": sample["idx"],
                            "prompt": sample["prompt"],
                            "prediction": prediction,
                            "prediction_processed": prediction_processed,
                            "prediction_finder": prediction_finder,
                            "label": sample["label"],
                        }
                    )
                save_json(
                    DATA_DIR_EVALUATION_SOCKET
                    / f"{task}/{self.model_id}_predictions_v2.json",
                    predictions,
                )

        else:
            raise ValueError("Task not recognized.")

    def _predict(self, sample) -> str:
        if self.use_inference_client:
            has_output = False
            while not has_output:
                try:
                    prediction = self.inference_client.text_generation(
                        sample["prompt"], **self.generation_kwargs
                    )
                except Exception:
                    time.sleep(2)
                    continue
                has_output = True

        else:
            prediction: str = self.llm(sample["prompt"])["generated_text"]
            prediction: str = prediction.replace(sample["prompt"], "")

        return prediction

    def _prepare_social_dim_test_data(self) -> List[Dict[str, str]]:
        """Prepare the test data for the social dimension task."""
        test_data: Dataset = self.social_dimensions.test_data

        test_data_formatted = {}

        # Loop through each JSON object and group by 'idx'
        for obj in test_data:
            idx = obj["idx"]
            response_good = obj["response_good"]

            if idx not in test_data_formatted:
                test_data_formatted[idx] = {
                    "label": [],
                    "idx": idx,
                    "prompt": self.social_dimensions._prompt_function(obj, is_q_a=True),
                }

            test_data_formatted[idx]["label"].append(response_good)

        # Return a list of all the values in the dictionary
        return list(test_data_formatted.values())

    def _prepare_socket_test_data(self, task: str) -> List[Dict[str, Union[str, int]]]:
        test_data_formatted: List[Dict[str, str]] = []

        # Get all the socket prompts with type CLS
        prompt = self.socket_prompts[self.socket_prompts["task"] == task][
            "question"
        ].iloc[0]

        dataset: Dataset = load_dataset("Blablablab/SOCKET", task, split="test")
        labels: List[str] = dataset.features["label"].names
        labels_formatted = [f'"{label}"' for label in labels]
        labels_mapping: Dict[int, str] = {i: label for i, label in enumerate(labels)}

        for idx, sample in enumerate(dataset):
            test_data_formatted.append(
                {
                    "idx": idx,
                    "prompt": self._prompt_socket(sample, prompt, labels_formatted),
                    "label": labels_mapping[sample["label"]],
                }
            )

        return test_data_formatted, labels

    def _prompt_socket(
        self, sample: Dict[str, str], prompt: str, labels: List[str]
    ) -> str:
        chat: List[Dict[str, str]] = self.llama_config.get_chat_template()

        chat[0]["content"] = chat[0]["content"].format(prompt_prefix="")

        task_prompt = (
            prompt.format(
                text=sample["text"],
            )
            + f" You can choose from the following labels: {', '.join(labels)}\nAnswer:"
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
    evaluator = Evaluator("meta-llama/Llama-2-7b-chat-hf")

    evaluator.predict(task="socket")

    a = 1
