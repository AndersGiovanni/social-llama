"""Evaluation script for the multilabel classification task. This is for the one-vs-rest approach."""
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from social_llama.config import DATA_DIR_MULTILABEL
from social_llama.utils import read_json


labels = [
    "Computer Science",
    "Physics",
    "Mathematics",
    "Statistics",
    "Quantitative Biology",
    "Quantitative Finance",
]

idx2label = {idx: label for idx, label in enumerate(labels)}
label2idx = {label: idx for idx, label in enumerate(labels)}

predictions = {}

for label in labels:
    preds = read_json(DATA_DIR_MULTILABEL / f"{label}.json")

    for idx, pred in enumerate(preds):
        if idx not in predictions:
            predictions[idx] = {
                "text": pred["text"],
                "predictions": [0] * len(labels),
                "true_labels": [0] * len(labels),
            }
            predictions[idx]["true_labels"][label2idx[label]] = pred["labels"]
            predictions[idx]["predictions"][label2idx[label]] = pred[label]

        else:
            predictions[idx]["true_labels"][label2idx[label]] = pred["labels"]
            predictions[idx]["predictions"][label2idx[label]] = pred[label]

preds = [predictions[idx]["predictions"] for idx in predictions]
trues = [predictions[idx]["true_labels"] for idx in predictions]

clf_report = classification_report(trues, preds, target_names=labels, output_dict=True)

accuracy = accuracy_score(trues, preds)
precision = precision_score(trues, preds, average="micro")
recall = recall_score(trues, preds, average="micro")
f1 = f1_score(trues, preds, average="micro")
hamming_loss_ = hamming_loss(trues, preds)

prediction_scores = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "hamming_loss": hamming_loss_,
    "classification_report": clf_report,
}

checkpoint = "meta-llama/Llama-2-7b-hf"

with open(
    DATA_DIR_MULTILABEL / f"{checkpoint}_ovr-model_multilabel-eval.json", "w"
) as f:
    json.dump(prediction_scores, f, indent=4)
