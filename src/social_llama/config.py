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
