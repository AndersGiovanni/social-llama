"""Define data paths and other configuration variables."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional


# Root directory of the project
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent

# Path to the data directory
DATA_DIR: Path = ROOT_DIR / "data"

# Subdirectories of the data directory
# Social Dimensions
DATA_DIR_SOCIAL_DIMENSIONS: Path = DATA_DIR / "social-dimensions"
DATA_DIR_SOCIAL_DIMENSIONS_RAW: Path = DATA_DIR_SOCIAL_DIMENSIONS / "raw"
DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED: Path = DATA_DIR_SOCIAL_DIMENSIONS / "processed"

# Evaluation
DATA_DIR_EVALUATION: Path = DATA_DIR / "evaluation"
DATA_DIR_EVALUATION_SOCKET: Path = DATA_DIR_EVALUATION / "socket"
DATA_DIR_EVALUATION_SOCIAL_DIMENSIONS: Path = DATA_DIR_EVALUATION / "social-dimensions"
DATA_DIR_EVALUATION_TEN_DIM: Path = DATA_DIR_EVALUATION / "ten-dim"

# Multilabel-test
DATA_DIR_MULTILABEL = DATA_DIR / "multilabel"
# Vector DB
DATA_DIR_VECTOR_DB: Path = DATA_DIR / "vector-db"
# Reverse Instructions
DATA_DIR_REVERSE_INSTRUCTIONS: Path = DATA_DIR / "reverse-instructions"


# Datasets Config Class
@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    pretty_name: str
    path: Path
    prompt_prefix: str
    prompt_template: str
    prompt_template_cot: str
    labels: List[str]
    max_generated_tokens: int
    cot_info_dict: Optional[Dict[str, str]]
    num_few_shot_examples: int = 0

    @property
    def id2label(self) -> Dict[int, str]:
        """Returns a dictionary mapping indices to labels.

        Returns:
            Dict[int, str]: Dictionary mapping indices to labels
        """
        return dict(enumerate(self.labels))

    @property
    def label2id(self) -> Dict[str, int]:
        """Returns a dictionary mapping labels to indices.

        Returns:
            Dict[str, int]: Dictionary mapping labels to indices
        """
        return {label: idx for idx, label in enumerate(self.labels)}


class Configs:
    """Instructions, templates, and other configurations for the Llama model."""

    default_llama_prompt: str = """<s>[INST] <<SYS>>
{system_message}
<</SYS>>

{user_message} [/INST]
"""
    default_system_message: str = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

{custom_message}
"""
    alternative_system_message: str = """You are an insightful and respectful assistant dedicated to fostering constructive social interaction, particularly on social media platforms.
Your primary objective is to analyze and unveil the core elements that enable spontaneous coordination and positive engagement in the face of social dilemmas.
Your responses should be well-informed, socially unbiased, respectful, positive, and honest, while promoting safety and constructive discourse.
You possess an extensive understanding of human interactions, with a keen focus on sparking collective coordination and meaningful debates.
Your knowledge should empower users to navigate social interactions in an informed and positive manner.
Moreover, your insights should reflect a deep understanding of the socio-cultural dynamics that underpin human communication on digital platforms, thereby promoting a culture of respect, understanding, and collective problem-solving.

{custom_message}"""

    def get_chat_template(self, system_role: str = "system") -> List[Dict[str, str]]:
        """Returns a chat template.

        Args:
            system_role (str): The role of the system in the chat. For Llama, it is 'system', and for Gemma it is 'model'.
        """
        chat: List[Dict[str, str]] = [
            {
                "role": system_role,
                # "content": """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n{prompt_prefix}""",
                # "content": """You are a helpful, respectful and honest assistant.""", # v1
                "content": """You are a helpful, respectful and honest assistant.\n{prompt_prefix}""",  # v2
            },
        ]
        return chat
