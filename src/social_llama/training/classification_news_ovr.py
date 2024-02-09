"""Train a model on the 10-dim dataset."""
import os
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from datasets import DatasetDict
from dotenv import load_dotenv
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import HfArgumentParser
from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainingArguments

import wandb
from social_llama.config import DATA_DIR
from social_llama.config import DATA_DIR_MULTILABEL
from social_llama.utils import save_json


load_dotenv()


@dataclass
class ScriptArguments:
    """Script arguments."""

    checkpoint: Optional[str] = field(
        default="roberta-large",
        metadata={
            "help": "the model name",
            "choices": [
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Llama-2-7b-hf",
                "mistralai/Mistral-7B-v0.1",
                "roberta-large",
                "bert-base-uncased",
            ],
        },
    )
    lr: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    num_train_epochs: Optional[int] = field(
        default=10, metadata={"help": "the number of training epochs"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "the per device train batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=4, metadata={"help": "the per device eval batch size"}
    )
    log_with: Optional[str] = field(
        default="wandb", metadata={"help": "use 'wandb' to log with wandb"}
    )
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "the logging frequency"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.01, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=2, metadata={"help": "the lora r parameter"})
    lora_bias: Optional[str] = field(
        default="none", metadata={"help": "the lora bias parameter"}
    )
    learning_rate: Optional[float] = field(
        default=1e-4, metadata={"help": "the learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="constant", metadata={"help": "the lr scheduler type"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.05, metadata={"help": "the number of warmup steps"}
    )
    output_dir: Optional[str] = field(
        default="./news-multilabel", metadata={"help": "the output directory"}
    )
    note: Optional[str] = field(
        default="binary", metadata={"help": "the note to add to the run"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


labels = [
    "Computer Science",
    "Physics",
    "Mathematics",
    "Statistics",
    "Quantitative Biology",
    "Quantitative Finance",
]

id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}


def preprocess_data(data, label: str):
    """Preprocess the data for one-vs-rest binary encoding based on the specified label.

    Args:
        data (DataFrame): The data to preprocess.
        label (str): The target label for one-vs-rest binary encoding.

    Returns:
        Dataset: The preprocessed data with binary encoding for the specified label.
    """
    task_labels = [
        "Computer Science",
        "Physics",
        "Mathematics",
        "Statistics",
        "Quantitative Biology",
        "Quantitative Finance",
    ]

    if label not in task_labels:
        raise ValueError(f"Label '{label}' not found in task labels.")

    texts = []
    binary_labels = []  # Changed from 'labels' to 'binary_labels' for clarity
    for _, item in data.iterrows():
        texts.append(f"{item['TITLE']}. {item['ABSTRACT']}".strip())
        # Create a binary label based on the presence of the specified label
        binary_labels.append(1 if item[label] == 1 else 0)

    # Create a dictionary with the texts and binary labels
    data_dict = {"text": texts, "labels": binary_labels}

    # Convert the dictionary to a Dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset


def split_data(data):
    """Split the data into train, validation, and test sets.

    Args:
        data (Dataset): The data to split.

    Returns:
        DatasetDict: The split data.
    """
    data = data.train_test_split(test_size=0.3, seed=42)  # 70% for train, 30% for test
    test = data["test"]
    test = test.train_test_split(
        test_size=0.5, seed=42
    )  # 15% for validation, 15% for test
    data["test"] = test["test"]
    data["validation"] = test["train"]

    # Convert the splits to a DatasetDict
    dataset_dict = DatasetDict(
        {"train": data["train"], "validation": data["validation"], "test": data["test"]}
    )

    return dataset_dict


def count_labels(dataset_dict, label):
    """Count the number of instances for each label.

    Args:
        dataset_dict (DatasetDict): The dataset to count the labels for.

    Returns:
        dict: The counts for each label.
    """
    label_counts = {
        "train": {"positive": 0, "negative": 0},
        "validation": {"positive": 0, "negative": 0},
        "test": {"positive": 0, "negative": 0},
    }

    # Iterate over each set in the dataset_dict
    for set_name in ["train", "validation", "test"]:
        # Iterate over the labels in the processed data
        for label_value in dataset_dict[set_name]["labels"]:
            if label_value == 1:  # If the label is present
                label_counts[set_name]["positive"] += 1
            else:  # If the label is not present
                label_counts[set_name]["negative"] += 1

    return label_counts


def calculate_weights(dataset_dict):
    """Calculate the weights the positive label.

    Args:
        dataset_dict (DatasetDict): The dataset to calculate the weights for.

    Returns:
        dict: The weights for each label.
    """
    pos_weights = len(dataset_dict["train"].to_pandas()) / (
        2 * dataset_dict["train"].to_pandas().labels.value_counts()[1]
    )
    neg_weights = len(dataset_dict["train"].to_pandas()) / (
        2 * dataset_dict["train"].to_pandas().labels.value_counts()[0]
    )

    return neg_weights, pos_weights


def compute_metrics(eval_pred):
    """Compute the metrics for the evaluation.

    Args:
        eval_pred (tuple): The evaluation predictions.

    Returns:
        dict: The metrics.
    """
    logits, labels = eval_pred
    # Convert logits to probabilities
    predictions = np.argmax(logits, axis=-1)
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="micro")
    recall = recall_score(labels, predictions, average="micro")
    f1 = f1_score(labels, predictions, average="micro")
    hamming_loss_ = hamming_loss(labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hamming_loss": hamming_loss_,
    }


def get_lora_model(model):
    """Get the LoRA model.

    Args:
        model (PreTrainedModel): The model to get the LoRA model for.

    Returns:
        PreTrainedModel: The LoRA model.
    """
    if script_args.checkpoint in [
        "HuggingFaceH4/zephyr-7b-beta",
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b-hf",
    ]:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            bias=script_args.lora_bias,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            bias=script_args.lora_bias,
        )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


class CustomCallback(TrainerCallback):
    """Custom callback to evaluate the model on the train set after each epoch."""

    def __init__(self, trainer) -> None:
        """Initialize the CustomCallback.

        Args:
            trainer (Trainer): The trainer to use for evaluation.

        Returns:
            None
        """
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate the model on the train set after each epoch.

        Args:
            args (TrainingArguments): The training arguments.
            state (TrainerState): The trainer state.
            control (TrainerControl): The trainer control.

        Returns:
            TrainerControl: The trainer control.
        """
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


class WeightedCELossTrainer(Trainer):
    """Trainer with weighted cross-entropy loss."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss.

        Args:
            model (PreTrainedModel): The model to compute the loss for.
            inputs (dict): The inputs to the model.
            return_outputs (bool, optional): Whether to return the outputs. Defaults to False.

        Returns:
        Union[float, tuple]: The loss or the loss and the outputs.
        """
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(
                [neg_weights, pos_weights], device=model.device, dtype=logits.dtype
            )
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def train_model(dataset_dict, model, tokenizer, label, test=False):
    """Train a model.

    Args:
        dataset_dict (DatasetDict): The dataset to train on.
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizer): The tokenizer used for the model.
        test (bool, optional): Whether to evaluate the model on the testset. Defaults to False.

    Returns:
        Trainer: The trainer used to train the model.
    """
    training_args = TrainingArguments(
        output_dir=f"{script_args.output_dir}/{label}/{script_args.checkpoint}",
        learning_rate=script_args.lr,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=script_args.log_with,
        save_total_limit=1,
        fp16=True
        if script_args.checkpoint
        in [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1",
        ]
        else False,
        gradient_checkpointing=script_args.gradient_checkpointing,
        run_name=f"{script_args.checkpoint}-{script_args.note}-{label}",
        seed=42,
        logging_dir=f"logs/multilabel-{label}-{script_args.checkpoint}",
    )

    # Define the data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, max_length=512
    )

    # Define the trainer
    trainer = WeightedCELossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Add Custom Callback
    trainer.add_callback(CustomCallback(trainer))

    # Train the model
    trainer.train()

    # Evaluate the model on the testset
    if test:
        trainer.evaluate(dataset_dict["test"], metric_key_prefix="test")

    return trainer


if __name__ == "__main__":
    # Set os.environ["WANDB_PROJECT"] to project name
    os.environ["WANDB_PROJECT"] = "multilabel-test"

    # Load script arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load the data
    data = pd.read_csv(DATA_DIR / "multilabel/train.csv")

    for label in labels:
        # Preprocess the data
        dataset = preprocess_data(data, label)

        # Split the data
        dataset_dict = split_data(dataset)

        # Count the labels
        label_counts = count_labels(dataset_dict, label)

        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=script_args.checkpoint,
            num_labels=2,
            trust_remote_code=True,
            device_map="auto",
        )

        # Calculate the weights
        neg_weights, pos_weights = calculate_weights(dataset_dict)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.checkpoint,
            trust_remote_code=True,
            truncation=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
        tokenizer.verbose = False

        # Set the pad token id
        model.config.pad_token_id = tokenizer.pad_token_id

        # Tokenize the data
        tokenized_datasets = dataset_dict.map(
            lambda examples: tokenizer(
                examples["text"],
                max_length=512,
                truncation=True,
                padding="max_length",
            ),
            batched=True,
            remove_columns=["text"],
        )

        # Set the format to torch
        tokenized_datasets.set_format("torch")

        # Fix size mismatch between model and tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # Get the LoRA model
        model = get_lora_model(model)

        # Train the model
        trainer = train_model(tokenized_datasets, model, tokenizer, label, test=True)

        # Save the model
        # trainer.save_model()

        predictions = trainer.predict(tokenized_datasets["test"])

        def softmax(x):
            """Compute the softmax of the input."""
            return np.exp(x) / sum(np.exp(x))

        save_outputs = []

        for sample, prediction_logits, true_label in zip(
            dataset_dict["test"], predictions.predictions, predictions.label_ids
        ):
            prediction = np.argmax(softmax(prediction_logits))
            sample[label] = int(prediction)
            sample[f"is_{label}"] = int(true_label)

            save_outputs.append(sample)

        save_json(DATA_DIR_MULTILABEL / f"{label}.json", save_outputs)

        wandb.finish()

        del model, trainer

        # Push to hub
        # trainer.push_to_hub()
