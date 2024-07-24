"""Calculate metrics for the evaluation of the socket data."""

import json
import os

# Suppress warnings
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

from social_llama.config import DATA_DIR_EVALUATION_SOCKET


warnings.filterwarnings("ignore")

tasks = [
    i
    for i in os.listdir(DATA_DIR_EVALUATION_SOCKET)
    if os.path.isdir(os.path.join(DATA_DIR_EVALUATION_SOCKET, i))
]

alias_mapping = {
    "hlab/SocialiteLlama_predictions_zero-shot.json": "Socialite",
    "meta-llama/Llama-2-7b-chat-hf_predictions_zero-shot.json": "Llama-2 Zero-Shot",
    "meta-llama/Meta-Llama-3-8B-Instruct_predictions_zero-shot.json": "Llama-3 Zero-Shot",
    "meta-llama/Llama-2-7b-chat-hf_predictions_RAG.json": "Llama-2 RAG",
    "meta-llama/Meta-Llama-3-8B-Instruct_predictions_knowledge.json": "Llama-3 Knowledge",
    "meta-llama/Llama-2-7b-chat-hf_predictions_knowledge.json": "Llama-2 Knowledge",
    "meta-llama/Meta-Llama-3-8B-Instruct_predictions_RAG.json": "Llama-3 RAG",
    "AndersGiovanni/social-llama-3-8b-instructions_predictions_zero-shot.json": "Social-Llama-3 Instructions",
    "AndersGiovanni/social-llama-7b-beta_predictions_zero-shot.json": "Social-Llama-2 Zero-Shot",
    "AndersGiovanni/social-llama-3-8b-beta_predictions_RAG.json": "Social-Llama-3 RAG",
    "AndersGiovanni/social-llama-7b-instructions_predictions_zero-shot.json": "Social-Llama-2 Instructions",
    "AndersGiovanni/social-llama-3-8b-instructions_predictions_RAG.json": "Social-Llama-3 Instructions RAG",
    "AndersGiovanni/social-llama-7b-instructions_predictions_RAG.json": "Social-Llama-2 Instructions RAG",
    "AndersGiovanni/social-llama-3-8b-beta_predictions_zero-shot.json": "Social-Llama-3 Zero-Shot",
    "AndersGiovanni/social-llama-7b-beta_predictions_RAG.json": "Social-Llama-2 RAG",
}

selected_tasks = [
    "complaints",
    "contextual-abuse#IdentityDirectedAbuse",
    "contextual-abuse#PersonDirectedAbuse",
    "crowdflower",
    "dailydialog",
    "empathy#distress_bin",
    "hahackathon#is_humor",
    "hasbiasedimplication",
    "hateoffensive",
    "hayati_politeness",
    "hypo-l",
    "implicit-hate#explicit_hate",
    "implicit-hate#implicit_hate",
    "implicit-hate#stereotypical_hate",
    "intentyn",
    "tweet_offensive",
    # "offensiveyn",
    "questionintimacy",
    "rumor#rumor_bool",
    "sarc",
    "stanfordpoliteness",
    "tweet_emotion",
    "tweet_irony",
    "tweet_offensive",
    "two-to-lie#receiver_truth",
]


results_llama = pd.DataFrame()

results_df = pd.DataFrame()


# Save results where
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
                or i.endswith("instruction_prompt.json")
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

            # Check if task row exists in the results dataframe
            if task not in results_df.columns:
                results_df[task] = pd.Series(dtype="float64")

            # If the f'model/{file}' column does not exist, create it and put it at the task row
            if f"{model}/{file}" not in results_df.index:
                results_df.loc[f"{model}/{file}"] = pd.Series(dtype="float64")
            results_df.at[f"{model}/{file}", task] = acc

            # Add mapping to the results dataframe
            if f"{model}/{file}" in alias_mapping:
                results_df.at[f"{model}/{file}", "alias"] = alias_mapping[
                    f"{model}/{file}"
                ]

            if f"{model}/{file}" not in results_df.index:
                results_df.loc[f"{model}/{file}"] = pd.Series(dtype="float64")
            results_df.at[f"{model}/{file}", task] = acc

            # Append model performance data for plotting
            model_names.append(model + "_" + file)
            accuracies.append(acc)

            # Append model performance data for storage
            if "knowledge.json" in file:
                column = "knowledge"
            elif "zero-shot.json" in file:
                if model == "hlab":
                    column = "sociallite-instrucitons-zero-shot"
                else:
                    if is_trained:
                        if "instructions" in file:
                            column = "reverse-instructions-zero-shot"
                        else:
                            column = "fine-tune-zero-shot"
                    else:
                        column = "zero-shot"
            elif "RAG.json" in file:
                if is_trained:
                    if "instructions" in file:
                        column = "reverse-instructions-RAG"
                    else:
                        column = "fine-tune-RAG"
                else:
                    column = "RAG"
            else:
                continue

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
                results_llama = pd.concat([results_llama, new_row], ignore_index=True)

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
    # plt.savefig(
    #     os.path.join(DATA_DIR_EVALUATION_SOCKET, "assets", f"{task}_accuracy.png")
    # )
    # plt.show()

# Save the results in a CSV
results_llama.to_csv(
    os.path.join(DATA_DIR_EVALUATION_SOCKET, "assets/results_llamas-total.csv"),
    index=False,
)

results_df.to_csv(
    os.path.join(DATA_DIR_EVALUATION_SOCKET, "assets/results_by_model.csv"), index=True
)
