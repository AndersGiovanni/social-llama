"""Evaluation metrics for the social dimensions task."""
import json
import os
from collections import Counter

from matplotlib import pyplot as plt

from social_llama.config import DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS


models = [
    i
    for i in os.listdir(DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS)
    if os.path.isdir(DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS) and i != "assets"
]

# Data storage for plotting
model_names = []
accuracies = []

# get all the files in the dir
for model in models:
    files = [
        i
        for i in os.listdir(os.path.join(DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS, model))
        if os.path.isfile(os.path.join(DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS, model, i))
    ]

    # get all the files in the subdir
    for file in files:
        if file.endswith(".json"):
            with open(
                os.path.join(DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS, model, file),
            ) as f:
                data = json.load(f)

                a = 1

        correct_preds = [
            p["prediction_processed"]
            for p in data
            if p["prediction_processed"] in p["label"]
        ]

        acc = len(correct_preds) / len(data)

        # Append model performance data for plotting
        model_names.append(model + "_" + file[:-20])
        accuracies.append(acc)

        # Print everything
        print(f"Model: {model}")
        print(f"File: {file}")
        print(f"Accuracy: {acc}")
        print(f"Counter: {Counter(correct_preds)}")
        print()

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.barh(model_names, accuracies)
# Adjust subplot left margin
plt.subplots_adjust(left=0.3)
plt.xlabel("Accuracy")
plt.title("Performance Metrics for Task: Social Dimensions")
plt.xlim(0, 1)  # Assuming accuracy is between 0 and 1
for bar in bars:
    plt.text(
        bar.get_width() + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{bar.get_width():.3f}",
        va="center",
        ha="left",
    )
plt.savefig(
    os.path.join(
        DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS,
        "assets",
        "social_dimensions_accuracy.png",
    )
)
