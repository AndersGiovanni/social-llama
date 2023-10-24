"""Evaluation of the model."""

from typing import List

import pandas as pd
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import pipeline

from social_llama.config import DATA_DIR_EVALUATION_SOCKET
from social_llama.config import LlamaConfigs
from social_llama.utils import get_device


class SocketEvaluator:
    """Evaluator for the Socket dataset."""

    def __init__(self, model_id: str) -> None:
        """Initialize the evaluator."""
        self.tasks: List[str] = ["CLS", "REG", "PAIR", "SPAN"]
        self.prompts: pd.DataFrame = pd.read_csv(
            DATA_DIR_EVALUATION_SOCKET / "socket_prompts.csv"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.config = AutoConfig.from_pretrained(model_id)
        self.llama_config = LlamaConfigs
        self.device = get_device()
        self.llm = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=self.tokenizer,
            device=self.device,
        )
