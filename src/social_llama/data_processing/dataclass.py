"""Dataclass to abstract the some data processing."""

from typing import Dict
from typing import List
from typing import Union

from datasets import Dataset
from datasets import DatasetDict
from torch.utils.data import Dataset as TorchDataset


class DataClass(TorchDataset):
    """Dataclass abstraction for the datasets. This gives us a unified framework."""

    def __init__(self) -> None:
        """Initialize the DataClass."""
        super().__init__()
        self.data: Union[DatasetDict, Dataset, None] = None
        self.labels: List[str] = []

    def set_data(self, data: Union[DatasetDict, Dataset]) -> None:
        """Sets the data.

        Args:
            data (Union[DatasetDict, Dataset]): Dataset or DatasetDict
        """
        self.data = data

    def preprocess(self) -> None:
        """This function should be overwritten by the child class.

        It should preprocess the data for the model.
        This includes tokenization, label preprocessing, and column formatting
        """
        raise NotImplementedError

    def label_to_idx_mapper(self) -> Dict[str, int]:
        """Returns a dictionary mapping labels to indices.

        Returns:
            Dict[str, int]: Dictionary mapping labels to indices
        """
        return {label: idx for idx, label in enumerate(self.labels)}

    def idx_to_label_mapper(self) -> Dict[int, str]:
        """Returns a dictionary mapping indices to labels.

        Returns:
            Dict[int, str]: Dictionary mapping indices to labels
        """
        return dict(enumerate(self.labels))

    def set_labels(self, labels: List[str]) -> None:
        """Sets the labels.

        Args:
            labels (List[str]): List of labels
        """
        self.labels = labels
