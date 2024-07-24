"""Weave evaluation for prediction files."""

import asyncio

# import json
import os

import weave
from huggingface_hub import InferenceClient

from social_llama.config import DATA_DIR_EVALUATION_SOCKET
from social_llama.utils import read_json


# from typing import Optional


# from weave.flow.scorer import MultiTaskBinaryClassificationF1


# from social_llama.utils import save_json


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

inference_client = InferenceClient(
    model=model_name, token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

some_pred_file = (
    DATA_DIR_EVALUATION_SOCKET
    / "contextual-abuse#IdentityDirectedAbuse/AndersGiovanni/social-llama-3-8b-beta_predictions_zero-shot.json"
)


# Extract the last part of the model name and remove .json
model_name = some_pred_file.stem
task_name = str(some_pred_file).split("/")[-3]
author_name = str(some_pred_file).split("/")[-2].replace("#", "-")

weave.init("social-llama-test")


class EvalPredictionFile(weave.Model):
    """Model for evaluating prediction files."""

    author: str
    model_name: str

    @weave.op()
    async def predict(self, sentence: str) -> dict:
        """Predict the target."""
        # parsed = json.loads(sentence)
        return {"prediction": sentence}


@weave.op()
def eval_sample(target: str, model_output: dict) -> dict:
    """Evaluate the prediction."""
    return {"correct": target == model_output["prediction"]}


data = read_json(some_pred_file)

# Change idx to id
for d in data:
    d["id"] = d.pop("idx")
    d["sentence"] = d.pop("prediction_finder")
    d["target"] = d.pop("label")

evaluation = weave.Evaluation(
    name=str(task_name),  # Task name
    dataset=data,
    scorers=[
        eval_sample,
    ],
)

model = EvalPredictionFile(model_name=model_name, author=author_name)

print(asyncio.run(evaluation.evaluate(model)))
