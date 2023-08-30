"""Define data paths and other configuration variables."""

from pathlib import Path


# Root directory of the project
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent

# Path to the data directory
DATA_DIR: Path = ROOT_DIR / "data"

# Subdirectories of the data directory
# Social Dimensions
DATA_DIR_SOCIAL_DIMENSIONS: Path = DATA_DIR / "social_dimensions"
DATA_DIR_SOCIAL_DIMENSIONS_RAW: Path = DATA_DIR_SOCIAL_DIMENSIONS / "raw"
DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED: Path = DATA_DIR_SOCIAL_DIMENSIONS / "processed"
