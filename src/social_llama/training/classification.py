import json
import os
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset
from datasets import DatasetDict
from dotenv import load_dotenv
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import HfArgumentParser
from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainingArguments

from social_llama.config import DATA_DIR_SOCIAL_DIMENSIONS_RAW


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
                "mistralai/Mistral-7B-v0.1",
                "roberta-large",
            ],
        },
    )
    lr: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    num_train_epochs: Optional[int] = field(
        default=10, metadata={"help": "the number of training epochs"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=8, metadata={"help": "the per device train batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=8, metadata={"help": "the per device eval batch size"}
    )
    log_with: Optional[str] = field(
        default="wandb", metadata={"help": "use 'wandb' to log with wandb"}
    )
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "the logging frequency"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the gradient accumulation steps"}
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
    lora_bias: Optional[str] = field(default='none', metadata={"help": "the lora bias parameter"})
    learning_rate: Optional[float] = field(
        default=1e-4, metadata={"help": "the learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.05, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.05, metadata={"help": "the weight decay"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )

    output_dir: Optional[str] = field(
        default="./ten-dim", metadata={"help": "the output directory"}
    )
    note: Optional[str] = field(
        default="", metadata={"help": "the note to add to the run"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


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
    "other",
]

int_2_label = {i: label for i, label in enumerate(labels)}
label_2_int = {label: i for i, label in enumerate(labels)}


def preprocess_data(data):
    # Extract the text and labels from the data
    texts = []
    labels = []
    for item in data:
        item_labels = [
            1 if item[key] >= 2 else 0
            for key in [
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
                "other",
            ]
        ]
        # Check if any label has a value of 2 or more
        texts.append(item["text"])
        labels.append(item_labels)

    # Create a dictionary with the texts and labels
    data_dict = {"text": texts, "labels": labels}

    # Convert the dictionary to a Dataset
    dataset = Dataset.from_dict(data_dict)

    return dataset


def split_data(data):
    # Split the data into train, validation, and test sets
    data = data.train_test_split(test_size=0.3, seed=42)  # 70% for train, 30% for test
    test = data["test"]
    test = test.train_test_split(
        test_size=0.5, seed=42
    )  # 10% for validation, 10% for test
    data["test"] = test["test"]
    data["validation"] = test["train"]

    # Convert the splits to a DatasetDict
    dataset_dict = DatasetDict(
        {"train": data["train"], "validation": data["validation"], "test": data["test"]}
    )

    return dataset_dict


def count_labels(dataset_dict):
    # Initialize a dictionary to store the counts for each set
    label_counts = {
        "train": {label: 0 for label in int_2_label.values()},
        "validation": {label: 0 for label in int_2_label.values()},
        "test": {label: 0 for label in int_2_label.values()},
    }

    # Iterate over each set in the dataset_dict
    for set_name in ["train", "validation", "test"]:
        # Iterate over the labels in the processed data
        for label_list in dataset_dict[set_name]["labels"]:
            for i, label_value in enumerate(label_list):
                if label_value == 1:  # If the label is present
                    label = int_2_label[i]  # Get the label name
                    label_counts[set_name][label] += 1  # Increment the count

    return label_counts


def calculate_weights(dataset_dict):
    # Calculate the total number of instances in the training set
    total_instances = len(dataset_dict["train"])

    # Calculate the number of instances for each label
    label_counts = {label: 0 for label in int_2_label.values()}
    for label_list in dataset_dict["train"]["labels"]:
        for i, label_value in enumerate(label_list):
            if label_value == 1:  # If the label is present
                label = int_2_label[i]  # Get the label name
                label_counts[label] += 1  # Increment the count

    # Calculate the weights for each label
    label_weights = {
        label: total_instances / (2 * count) for label, count in label_counts.items()
    }

    return label_weights


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities
    probs = expit(logits)
    # Use a threshold to convert probabilities to binary predictions
    predictions = (probs > 0.5).astype(int)
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="micro")
    recall = recall_score(labels, predictions, average="micro")
    f1 = f1_score(labels, predictions, average="micro")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_lora_model(model):
    if (
        script_args.checkpoint == "mistralai/Mistral-7B-v0.1"
        or script_args.checkpoint == "meta-llama/Llama-2-7b-hf"
    ):
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
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Convert label weights to tensor
        weights = torch.tensor(
            [label_weights[label] for label in int_2_label.values()],
            device=logits.device,
            dtype=logits.dtype,
        )
        # Compute custom loss
        loss_fct = torch.nn.BCEWithLogitsLoss(weight=weights)
        labels = labels.type_as(logits)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_model(dataset_dict, model, tokenizer, test=False):
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=f"{script_args.output_dir}/{script_args.checkpoint}",
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
        # fp16=True,
        gradient_checkpointing=script_args.gradient_checkpointing,
        run_name=f"{script_args.checkpoint}-{script_args.note}",
        seed=42,
        logging_dir=f"logs/10dim-{script_args.checkpoint}",
    )

    # Define the data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, max_length=model.config.max_position_embeddings
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
    os.environ["WANDB_PROJECT"] = "ten-dim"

    # Load script arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load the data
    data = json.load(open(DATA_DIR_SOCIAL_DIMENSIONS_RAW / "labeled_dataset.json"))

    # Preprocess the data
    dataset = preprocess_data(data)

    # Split the data
    dataset_dict = split_data(dataset)

    # Count the labels
    label_counts = count_labels(dataset_dict)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=script_args.checkpoint,
        num_labels=11,
        trust_remote_code=True,
        problem_type="multi_label_classification",
        # device_map="auto",
    )
    
    # Calculate the weights
    label_weights = calculate_weights(dataset_dict)

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

    # Tokenize the data
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=model.config.max_position_embeddings,
        ),
        batched=True,
        remove_columns=["text"],
    )
    tokenized_datasets.set_format("torch")

    # Fix size mismatch between model and tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Get the LoRA model
    model = get_lora_model(model)

    # Train the model
    trainer = train_model(tokenized_datasets, model, tokenizer, test=True)

    # Save the model
    # trainer.save_model()
