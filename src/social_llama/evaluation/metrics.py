"""Calculate metrics for the evaluation of the socket data."""

import json
import os

# Suppress warnings
import warnings

# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

from social_llama.config import DATA_DIR_EVALUATION_SOCKET


warnings.filterwarnings("ignore")

tasks = [
    i
    for i in os.listdir(DATA_DIR_EVALUATION_SOCKET)
    if os.path.isdir(os.path.join(DATA_DIR_EVALUATION_SOCKET, i))
]

selected_tasks = [
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
]


results_llama = pd.DataFrame(
    columns=[
        "task",
        "zero-shot",
        "knowledge",
        "RAG",
        "fine-tune-zero-shot",
        "fine-tune-knowledge",
        "fine-tune-RAG",
    ]
)
results_gemma = pd.DataFrame(
    columns=[
        "task",
        "zero-shot",
        "knowledge",
        "RAG",
        "fine-tune-zero-shot",
        "fine-tune-knowledge",
        "fine-tune-RAG",
    ]
)

for task in tasks:
    if task not in selected_tasks:
        continue
    print("-" * 50)
    print(f"Task: {task}")

    models = [
        i
        for i in os.listdir(os.path.join(DATA_DIR_EVALUATION_SOCKET, task))
        if os.path.isdir(os.path.join(DATA_DIR_EVALUATION_SOCKET, task, i))
    ]

    if "AGMoller" in models:
        models.remove("AGMoller")

    # Data storage for plotting
    model_names = []
    accuracies = []

    for model in models:
        if model in ["google", "meta-llama"]:
            is_trained = False
        else:
            is_trained = True

        files = [
            i
            for i in os.listdir(os.path.join(DATA_DIR_EVALUATION_SOCKET, task, model))
            if os.path.isfile(os.path.join(DATA_DIR_EVALUATION_SOCKET, task, model, i))
            and (
                i.endswith("knowledge.json")
                or i.endswith("zero-shot.json")
                or i.endswith("RAG.json")
            )
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

            # Append model performance data for storage
            if "knowledge.json" in file:
                column = "fine-tune-knowledge" if is_trained else "knowledge"
            elif "zero-shot.json" in file:
                column = "fine-tune-zero-shot" if is_trained else "zero-shot"
            elif "RAG.json" in file:
                column = "fine-tune-RAG" if is_trained else "RAG"
            else:
                continue

            # Append model performance data for storage
            if model == "google":
                if task in results_gemma["task"].values:
                    results_gemma.loc[results_gemma["task"] == task, column] = acc
                else:
                    new_row = pd.DataFrame(
                        {
                            "task": [task],
                            column: [acc],
                            **{
                                col: [None]
                                for col in results_gemma.columns
                                if col not in ["task", column]
                            },
                        }
                    )
                    results_gemma = pd.concat(
                        [results_gemma, new_row], ignore_index=True
                    )
            else:
                if task in results_llama["task"].values:
                    results_llama.loc[results_llama["task"] == task, column] = acc
                else:
                    new_row = pd.DataFrame(
                        {
                            "task": [task],
                            column: [acc],
                            **{
                                col: [None]
                                for col in results_llama.columns
                                if col not in ["task", column]
                            },
                        }
                    )
                    results_llama = pd.concat(
                        [results_llama, new_row], ignore_index=True
                    )

    if len(model_names) <= 1:
        continue

    # Create a bar chart
    # plt.figure(figsize=(20, 10))
    # colors = [
    #     (
    #         "#06d6a0"
    #         if "knowledge" in model
    #         else (
    #             "#ef476f"
    #             if "knwldg" in model
    #             else "#ffd166"
    #             if "RAG" in model
    #             else "#118ab2"
    #         )
    #     )
    #     for model in model_names
    # ]
    # bars = plt.barh(model_names, accuracies, color=colors)
    # # Adjust subplot left margin
    # plt.subplots_adjust(left=0.4)
    # plt.xlabel("Accuracy")
    # plt.title(f"Performance Metrics for Task: {task}")
    # plt.xlim(0, 1)  # Assuming accuracy is between 0 and 1
    # for bar in bars:
    #     plt.text(
    #         bar.get_width() + 0.01,
    #         bar.get_y() + bar.get_height() / 2,
    #         f"{bar.get_width():.3f}",
    #         va="center",
    #         ha="left",
    #     )
    # plt.savefig(
    #     os.path.join(DATA_DIR_EVALUATION_SOCKET, "assets", f"{task}_accuracy.png")
    # )
    # plt.show()

# Save the results in a CSV
results_llama.to_csv(
    os.path.join(DATA_DIR_EVALUATION_SOCKET, "assets/results_llama.csv"), index=False
)
results_gemma.to_csv(
    os.path.join(DATA_DIR_EVALUATION_SOCKET, "assets/results_gemma.csv"), index=False
)
