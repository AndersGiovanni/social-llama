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
    if task not in [
        "hasbiasedimplication",
        "implicit-hate#stereotypical_hate",
        "intentyn",
        "tweet_offensive",
        "offensiveyn",
        "empathy#distress_bin",
        "complaints",
        "hayati_politeness",
        "stanfordpoliteness",
        "hypo-l",
        "rumor#rumor_bool",
        "two-to-lie#receiver_truth",
        "hahackathon#is_humor",
        "sarc",
        "contextual-abuse#IdentityDirectedAbuse",
        "contextual-abuse#PersonDirectedAbuse",
        "tweet_irony",
        "questionintimacy",
        "tweet_emotion",
        "hateoffensive",
        "implicit-hate#explicit_hate",
        "implicit-hate#implicit_hate",
        "crowdflower",
        "dailydialog",
    ]:
        continue
    print("-" * 50)
    print(f"Task: {task}")

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

            # predictions_processed = [p["prediction_processed"] for p in data]
            # predictions_finder = [p["prediction_finder"] for p in data]
            correct_key = (
                "prediction_finder" if "prediction_finder" in data[0] else "prediction"
            )
            predictions = [p[correct_key] for p in data]
            labels = [p["label"] for p in data]

            acc = accuracy_score(predictions, labels)

            # Append model performance data for plotting
            model_names.append(model + "_" + file)
            accuracies.append(acc)

    if len(model_names) <= 1:
        continue

    # Create a bar chart
    plt.figure(figsize=(20, 10))
    colors = [
        (
            "#06d6a0"
            if "knowledge" in model
            else (
                "#ef476f"
                if "knwldg" in model
                else "#ffd166"
                if "RAG" in model
                else "#118ab2"
            )
        )
        for model in model_names
    ]
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
