import json
from typing import List
from typing import Union

import datasets
import torch
from datasets import Dataset
from datasets import DatasetDict
from datasets import IterableDataset
from datasets import IterableDatasetDict
from peft import PeftConfig
from peft import PeftModel
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast

from social_llama.config import DATA_DIR_EVALUATION_TEN_DIM


labels = [
    "social_support",
    "conflict",
    "trust",
    "fun",
    "similarity",
    "identity",
    "respect",
    "romance",
    "knowledge",
    "power",
]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}


def get_model(model_name):
    """Load a model from the HuggingFace Hub.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        Tuple[PeftModel, AutoTokenizer]: The loaded PEFT model and tokenizer.

    """
    peft_model_id = model_name
    config = PeftConfig.from_pretrained(peft_model_id)
    inference_model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path,
        num_labels=10,
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    tokenizer.verbose = False

    # Set the pad token id

    model = PeftModel.from_pretrained(inference_model, peft_model_id)
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def evaluate_model(
    dataset_dict: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    model_name: str,
    batch_size: int = 4,
):
    """Evaluate a model on a dataset and save the results to a JSON file.

    Args:
        dataset_dict (Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]): The dataset to evaluate on.
        model (PreTrainedModel): The model to evaluate.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer used for the model.
        model_name (str): The name of the model.
        batch_size (int, optional): The batch size to use for evaluation. Defaults to 4.

    Returns:
        None
    """

    # Step 2: Prepare the test data
    test_texts: List[str] = dataset_dict["text"]  # type: ignore
    test_labels: List[List[str]] = dataset_dict["labels"]  # type: ignore

    predictions = []

    # Step 3: Get predictions
    for i in tqdm(range(0, len(test_texts), batch_size)):
        batch_texts = test_texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="longest",
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs).logits
        probs = expit(outputs)
        binary_predictions = (probs > 0.5).to(torch.int)
        predictions.extend(binary_predictions.tolist())

    # Compute metrics
    accuracy = accuracy_score(test_labels, predictions)
    clf_report = classification_report(
        test_labels, predictions, target_names=labels, output_dict=True
    )
    clf_report["accuracy"] = accuracy
    cm = multilabel_confusion_matrix(test_labels, predictions)

    # Prepare data for JSON
    results = {
        "clf_report": clf_report,
        "confusion_matrix": {
            label: cm[i].tolist() for i, label in enumerate(labels)
        },  # numpy arrays are not serializable
        "predictions": [
            {
                "text": text,
                "labels": [
                    id2label[i] for i, label in enumerate(labels) if label == 1
                ],  # convert label indices to label strings,  # convert label indices to label strings
                "predicted_labels": [
                    id2label[i] for i, pred in enumerate(pred) if pred == 1
                ],  # convert predicted label indices to label strings
            }
            for text, pred, labels in zip(test_texts, predictions, test_labels)
        ],
    }

    # Save to JSON file
    with open(
        DATA_DIR_EVALUATION_TEN_DIM / f"{model_name.split('/')[-1]}.json", "w"
    ) as f:
        json.dump(results, f)


data = datasets.load_dataset("AndersGiovanni/10-dim", split="test")

models = [
    # "AndersGiovanni/roberta-large-lora-10-dim",
    "AndersGiovanni/Mistral-7B-v0.1-lora-10-dim",
    "AndersGiovanni/Llama-2-7b-hf-lora-10-dim",
]


for model_name in models:
    model, tokenizer = get_model(model_name)

    evaluate_model(data, model, tokenizer, model_name)

    del model
