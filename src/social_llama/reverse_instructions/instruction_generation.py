"""Generate reverse instructions for the socket benchmark."""

from typing import Dict
from typing import List

import pandas as pd
from datasets import load_dataset
from openai import OpenAI
from torch.utils.data import DataLoader
from tqdm import tqdm

from social_llama.config import DATA_DIR_EVALUATION_SOCKET
from social_llama.config import DATA_DIR_REVERSE_INSTRUCTIONS
from social_llama.reverse_instructions.instruction_configs import (
    ReverseInstructionsPrompts,
)
from social_llama.reverse_instructions.utils import estimate_total_costs_from_sample
from social_llama.utils import save_json


# Get all the tasks
socket_prompts: pd.DataFrame = pd.read_csv(
    DATA_DIR_EVALUATION_SOCKET / "socket_prompts.csv"
)

# Get all classification tasks
cls_tasks = socket_prompts[socket_prompts["type"] == "CLS"].head(5)

task_data = {}

for task in tqdm(cls_tasks["task"].unique(), desc="Load and sample data"):
    # Load the dataset for the task
    dataset = load_dataset(
        "Blablablab/SOCKET",
        task,
        trust_remote_code=True,
        num_proc=8,
    )

    # Remove the sockette split
    if "sockette" in dataset:
        del dataset["sockette"]

    select_size = 10

    # Sample 2000 examples from the 'train' split of the dataset
    if len(dataset["train"]) > select_size:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(select_size))

    # Sample 10% of the 'train' split size from the other splits
    sample_size = len(dataset["train"]) // 10
    for split in dataset.keys():
        if split != "train" and len(dataset[split]) > sample_size:
            dataset[split] = dataset[split].shuffle(seed=42).select(range(sample_size))

    # Store the dataset in the dictionary
    task_data[task] = dataset

# Generate the reverse instructions

# Initialize the reverse instructions prompts
(
    system_prompt,
    reverse_instructions_prompts,
) = ReverseInstructionsPrompts().reverse_instruction_cls()

client = OpenAI()


for task, dataset in tqdm(
    task_data.items(),
    desc="Generate reverse instructions",
    total=len(task_data),
    unit="task:",
):
    task_data_reverse_instructions = {}

    for split, data in tqdm(dataset.items(), desc=f"Task: {task}", unit="split"):
        labels: List[str] = data.features["label"].names
        labels_mapping: Dict[int, str] = {i: label for i, label in enumerate(labels)}

        for sample in tqdm(DataLoader(data, batch_size=1), desc=f"Split: {split}"):
            # Prepare the prompt
            text: str = sample["text"][0]
            label: str = labels_mapping[sample["label"].item()]

            sample_reverse_instruction_prompt = reverse_instructions_prompts.format(
                text=text, label_list=labels, label=label
            )

            # Using default temperature and top_p (1)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sample_reverse_instruction_prompt},
                ],
            )

            # Store the reverse instruction
            sample_output = {
                "text": text,
                "label": label,
                "prompt": sample_reverse_instruction_prompt,
                "reverse_instruction": response.choices[0].message.content,
                "metadata": {
                    "created": response.created,
                    "model": response.model,
                    "usage": response.usage.__dict__,
                },
            }

            task_data_reverse_instructions.setdefault(split, []).append(sample_output)

        # Save the reverse instructions
    save_json(
        DATA_DIR_REVERSE_INSTRUCTIONS / f"{task}_reverse_instructions.json",
        [task_data_reverse_instructions],
    )

    price_per_million_completion_tokens = 1.5  # Adjust as per the actual price
    price_per_million_prompt_tokens = 0.5  # Adjust as per the actual price
    num_samples = 2400  # The number of samples you intend to have

    # Estimate the total costs
    estimated_costs = estimate_total_costs_from_sample(
        [task_data_reverse_instructions],
        price_per_million_completion_tokens,
        price_per_million_prompt_tokens,
        num_samples,
    )
    print(f"Estimated costs for {task}: {estimated_costs}")
