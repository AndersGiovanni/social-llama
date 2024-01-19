import json
from copy import deepcopy

import numpy as np
import torch
from datasets import Dataset
from datasets import DatasetDict
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
from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainingArguments

from social_llama.config import DATA_DIR_SOCIAL_DIMENSIONS_RAW


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


data = json.load(open(DATA_DIR_SOCIAL_DIMENSIONS_RAW / "labeled_dataset.json"))

data = preprocess_data(data)

# Split the data into train, validation, and test sets
data = data.train_test_split(test_size=0.3, seed=42)  # 80% for train, 20% for test
test = data["test"]
test = test.train_test_split(test_size=0.5, seed=42)  # 10% for validation, 10% for test
data["test"] = test["test"]
data["validation"] = test["train"]

# Convert the splits to a DatasetDict
dataset_dict = DatasetDict(
    {"train": data["train"], "validation": data["validation"], "test": data["test"]}
)

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

# Print the counts for each set
for set_name, counts in label_counts.items():
    print(f"{set_name} counts: {counts}")

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

# Print the weights for each label
for label, weight in label_weights.items():
    print(f"{label} weight: {weight}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to probabilities
    probs = expit(logits)
    # Use a threshold to convert probabilities to binary predictions
    predictions = (probs > 0.5).astype(int)

    # Compute metrics
    precision = precision_score(labels, predictions, average="micro")
    recall = recall_score(labels, predictions, average="micro")
    f1 = f1_score(labels, predictions, average="micro")
    accuracy = accuracy_score(labels, predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "accuracy": accuracy,
    }


class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Convert label weights to tensor
        weights = torch.tensor(
            [label_weights[label] for label in int_2_label.values()],
            device=model.device,
        )
        # Compute custom loss
        loss_fct = torch.nn.BCEWithLogitsLoss(weight=weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Load Mistral 7B Tokenizer
mistral_checkpoint = "mistralai/Mistral-7B-v0.1"
mistral_tokenizer = AutoTokenizer.from_pretrained(
    mistral_checkpoint, add_prefix_space=True
)
mistral_tokenizer.pad_token_id = mistral_tokenizer.eos_token_id
mistral_tokenizer.pad_token = mistral_tokenizer.eos_token


def mistral_preprocessing_function(examples):
    return mistral_tokenizer(examples["text"], truncation=True, max_length=1024)


mistral_tokenized_datasets = data.map(
    mistral_preprocessing_function, batched=True, remove_columns=["text"]
)
# mistral_tokenized_datasets = mistral_tokenized_datasets.rename_column("target", "label")
mistral_tokenized_datasets.set_format("torch")

# Data collator for padding a batch of examples to the maximum length seen in the batch
mistral_data_collator = DataCollatorWithPadding(tokenizer=mistral_tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=mistral_checkpoint,
    num_labels=10,
    trust_remote_code=True,
    problem_type="multi_label_classification",
    device_map="auto",
)

model.config.pad_token_id = model.config.eos_token_id


mistral_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=2,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

mistral_model = get_peft_model(model, mistral_peft_config)
mistral_model.print_trainable_parameters()

mistral_model = mistral_model.cuda()

lr = 1e-4
batch_size = 8
num_epochs = 5

training_args = TrainingArguments(
    output_dir="mistral-lora-token-classification",
    learning_rate=lr,
    lr_scheduler_type="constant",
    warmup_ratio=0.1,
    max_grad_norm=0.3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    fp16=True,
    gradient_checkpointing=True,
)


mistral_trainer = WeightedCELossTrainer(
    model=mistral_model,
    args=training_args,
    train_dataset=mistral_tokenized_datasets["train"],
    eval_dataset=mistral_tokenized_datasets["validation"],
    data_collator=mistral_data_collator,
    compute_metrics=compute_metrics,
)


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


mistral_trainer.add_callback(CustomCallback(mistral_trainer))

mistral_trainer.train()