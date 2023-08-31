"""Testing the DataClass."""

import unittest

from social_llama.config import DatasetConfig
from social_llama.data_processing.dataclass import DataClass


class TestDataClass(unittest.TestCase):
    """Test cases for the DataClass."""

    def setUp(self) -> None:
        """Setup the test case."""
        mock_config = DatasetConfig(
            name="mock",
            pretty_name="Mock",
            prompt_prefix="mock",
            prompt_template="mock",
            labels=["label1", "label2", "label3"],
            max_generated_tokens=10,
        )
        self.data_class = DataClass(mock_config)
        self.labels = ["label1", "label2", "label3"]

    def test_preprocess_not_implemented(self) -> None:
        """Test the preprocess method."""
        with self.assertRaises(NotImplementedError):
            self.data_class.preprocess()
