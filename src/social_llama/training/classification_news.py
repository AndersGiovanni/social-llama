"""Train a model on the 10-dim dataset."""
import os
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset
from datasets import DatasetDict
from dotenv import load_dotenv
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
from scipy.special import expit
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

from social_llama.config import DATA_DIR
from social_llama.config import DATA_DIR_MULTILABEL
from social_llama.utils import save_json


load_dotenv()


@dataclass
class ScriptArguments:
    """Script arguments."""

    checkpoint: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf",
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
        default="", metadata={"help": "the note to add to the run"}
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


def preprocess_data(data):
    """Preprocess the data.

    Args:
        data (list): The data to preprocess.

    Returns:
        Dataset: The preprocessed data.
    """
    task_labels = [
        "Computer Science",
        "Physics",
        "Mathematics",
        "Statistics",
        "Quantitative Biology",
        "Quantitative Finance",
    ]

    texts = []
    labels = []
    for _, item in data.iterrows():
        texts.append(f"{item['TITLE']}. {item['ABSTRACT']}".strip())
        labels.append(item[task_labels].tolist())

    # Create a dictionary with the texts and labels
    data_dict = {"text": texts, "labels": labels}

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


def count_labels(dataset_dict):
    """Count the number of instances for each label.

    Args:
        dataset_dict (DatasetDict): The dataset to count the labels for.

    Returns:
        dict: The counts for each label.
    """
    label_counts = {
        "train": {label: 0 for label in id2label.values()},
        "validation": {label: 0 for label in id2label.values()},
        "test": {label: 0 for label in id2label.values()},
    }

    # Iterate over each set in the dataset_dict
    for set_name in ["train", "validation", "test"]:
        # Iterate over the labels in the processed data
        for label_list in dataset_dict[set_name]["labels"]:
            for i, label_value in enumerate(label_list):
                if label_value == 1:  # If the label is present
                    label = id2label[i]  # Get the label name
                    label_counts[set_name][label] += 1  # Increment the count

    return label_counts


def calculate_weights(dataset_dict):
    """Calculate the weights for each label.

    Args:
        dataset_dict (DatasetDict): The dataset to calculate the weights for.

    Returns:
        dict: The weights for each label.
    """
    total_instances = len(dataset_dict["train"])

    # Calculate the number of instances for each label
    label_counts = {label: 0 for label in id2label.values()}
    for label_list in dataset_dict["train"]["labels"]:
        for i, label_value in enumerate(label_list):
            if label_value == 1:  # If the label is present
                label = id2label[i]  # Get the label name
                label_counts[label] += 1  # Increment the count

    # Calculate the weights for each label
    label_weights = {
        label: total_instances / (2 * count) for label, count in label_counts.items()
    }

    return label_weights


def compute_metrics(eval_pred):
    """Compute the metrics for the evaluation.

    Args:
        eval_pred (tuple): The evaluation predictions.

    Returns:
        dict: The metrics.
    """
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
        # Convert label weights to tensor
        weights = torch.tensor(
            [label_weights[label] for label in id2label.values()],
            device=logits.device,
            dtype=logits.dtype,
        )
        # Compute custom loss
        loss_fct = torch.nn.BCEWithLogitsLoss(weight=weights)
        labels = labels.type_as(logits)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_model(dataset_dict, model, tokenizer, test=False):
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
        fp16=True
        if script_args.checkpoint
        in [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1",
        ]
        else False,
        gradient_checkpointing=script_args.gradient_checkpointing,
        run_name=f"{script_args.checkpoint}-{script_args.note}",
        seed=42,
        logging_dir=f"logs/multilabel-{script_args.checkpoint}",
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

    # Preprocess the data
    dataset = preprocess_data(data)

    # Split the data
    dataset_dict = split_data(dataset)

    # Count the labels
    label_counts = count_labels(dataset_dict)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=script_args.checkpoint,
        num_labels=len(labels),
        trust_remote_code=True,
        problem_type="multi_label_classification",
        device_map="auto",
        id2label=id2label,
        label2id=label2id,
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
    trainer = train_model(tokenized_datasets, model, tokenizer, test=True)

    # Save the model
    # trainer.save_model()

    predictions = trainer.predict(tokenized_datasets["test"])

    predictions_per_class = {
        label: {
            "pred": [],
            "true": [],
        }
        for label in id2label.values()
    }

    for prediction_logits, true_label in zip(
        predictions.predictions, predictions.label_ids
    ):
        for i, label in enumerate(id2label.values()):
            probs = expit(prediction_logits[i])
            predictions_per_class[label]["pred"].append((probs > 0.5).astype(int))
            predictions_per_class[label]["true"].append(true_label[i])

    metrics = {}

    # Save metrics to json file
    for label, preds in predictions_per_class.items():
        accuracy = accuracy_score(preds["true"], preds["pred"])
        precision = precision_score(preds["true"], preds["pred"], average="micro")
        recall = recall_score(preds["true"], preds["pred"], average="micro")
        f1 = f1_score(preds["true"], preds["pred"], average="micro")
        hamming_loss_ = hamming_loss(preds["true"], preds["pred"])
        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hamming_loss": hamming_loss_,
        }
        metrics[label] = scores

    save_json(
        DATA_DIR_MULTILABEL
        / f"{script_args.checkpoint}_multilabel-model_ovr-eval.json",
        metrics,
    )

    # Push to hub
    # trainer.push_to_hub()
