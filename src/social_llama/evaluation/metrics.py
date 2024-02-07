"""Calculate metrics for the evaluation of the socket data."""

import json
import os

# Suppress warnings
import warnings

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from social_llama.config import DATA_DIR_EVALUATION_SOCKET


warnings.filterwarnings("ignore")

tasks = [
    i
    for i in os.listdir(DATA_DIR_EVALUATION_SOCKET)
    if os.path.isdir(os.path.join(DATA_DIR_EVALUATION_SOCKET, i))
]

for task in tasks:
    print("-" * 50)
    print(f"Task: {task}")

    if not task in [
        "contextual-abuse#PersonDirectedAbuse",
        "contextual-abuse#IdentityDirectedAbuse",
        "tweet_irony",
        "hateoffensive",
        "tweet_emotion",
        "implicit-hate#explicit_hate",
        "implicit-hate#implicit_hate",
        "crowdflower",
    ]:
        continue

    models = [
        i
        for i in os.listdir(os.path.join(DATA_DIR_EVALUATION_SOCKET, task))
        if os.path.isdir(os.path.join(DATA_DIR_EVALUATION_SOCKET, task, i))
    ]

    # Data storage for plotting
    model_names = []
    accuracies = []

    for model in models:
        files = [
            i
            for i in os.listdir(os.path.join(DATA_DIR_EVALUATION_SOCKET, task, model))
            if os.path.isfile(os.path.join(DATA_DIR_EVALUATION_SOCKET, task, model, i))
        ]

        for file in files:
            if file.endswith(".json"):
                with open(
                    os.path.join(DATA_DIR_EVALUATION_SOCKET, task, model, file)
                ) as f:
                    data = json.load(f)

            predictions_processed = [p["prediction_processed"] for p in data]
            predictions_finder = [p["prediction_finder"] for p in data]
            labels = [p["label"] for p in data]

            acc = accuracy_score(predictions_finder, labels)

            # Append model performance data for plotting
            model_names.append(model + "_" + file)
            accuracies.append(acc)

    if len(model_names) <= 1:
        continue

    # Create a bar chart
    plt.figure(figsize=(20, 10))
    colors = [
        "red" if "knowledge" in model or "knwldg" in model else "blue"
        for model in model_names
    ]
    colors = []
    for model in model_names:
        if "knowledge" in model:
            colors.append("#06d6a0")
        elif "knwldg" in model:
            colors.append("#ef476f")
        else:
            colors.append("#118ab2")
    bars = plt.barh(model_names, accuracies, color=colors)
    # Adjust subplot left margin
    plt.subplots_adjust(left=0.4)
    plt.xlabel("Accuracy")
    plt.title(f"Performance Metrics for Task: {task}")
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
        os.path.join(DATA_DIR_EVALUATION_SOCKET, "assets", f"{task}_accuracy.png")
    )
    # plt.show()
