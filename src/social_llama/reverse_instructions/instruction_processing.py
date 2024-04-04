"""Process and clean the raw data for the reverse instructions task."""

import os

from datasets import load_dataset

from social_llama.config import DATA_DIR_REVERSE_INSTRUCTIONS
from social_llama.utils import read_json
from social_llama.utils import save_json


def clean_raw_data():
    """Clean the raw data for the reverse instructions task."""
    files = os.listdir(DATA_DIR_REVERSE_INSTRUCTIONS / "raw")

    skips = 0

    for file in files:
        clean_file = file.replace("_reverse_instructions.json", "")

        data = read_json(DATA_DIR_REVERSE_INSTRUCTIONS / "raw" / file)

        dataset = load_dataset(
            "Blablablab/SOCKET",
            clean_file,
            trust_remote_code=True,
            num_proc=8,
        )

        labels_list = dataset["train"].features["label"].names

        processed_data = {}

        for split, samples in data[0].items():
            if split == "generation_costs":
                continue

            processed_data[split] = []

            for sample in samples:
                # Below are some basic heuristics to clean the data a bit

                # If the text is in the reverse instruction, skip
                if sample["text"] in sample["reverse_instruction"]:
                    skips += 1
                    continue

                # If the label is in the reverse instruction, skip
                if "the correct label" in sample["reverse_instruction"]:
                    skips += 1
                    continue

                # If the API call failed, skip
                if sample["reverse_instruction"][:7] == "Failed.":
                    skips += 1
                    continue

                if sample["reverse_instruction"][:3] != "X: ":
                    skips += 1
                    continue

                processed_data[split].append(
                    {
                        "text": sample["text"],
                        "label": sample["label"],
                        "instruction": sample["reverse_instruction"][3:],
                        "task": clean_file,
                        "label_options": labels_list,
                    }
                )

        save_json(
            DATA_DIR_REVERSE_INSTRUCTIONS / "processed" / clean_file / "train.json",
            processed_data["train"],
        )

        save_json(
            DATA_DIR_REVERSE_INSTRUCTIONS
            / "processed"
            / clean_file
            / "validation.json",
            processed_data["validation"],
        )
        save_json(
            DATA_DIR_REVERSE_INSTRUCTIONS / "processed" / clean_file / "test.json",
            processed_data["test"],
        )
        # based on a list of strings, make a "labels_list.txt" file with one label per line
        with open(
            DATA_DIR_REVERSE_INSTRUCTIONS
            / "processed"
            / clean_file
            / "labels_list.txt",
            "w",
        ) as f:
            for label in labels_list:
                f.write(f"{label}\n")

    print(f"Skipped {skips} samples.")


# clean_raw_data()

dataset = load_dataset(
    "AndersGiovanni/instructions-SOCKET", num_proc=1, download_mode="force_redownload"
)

a = 1
