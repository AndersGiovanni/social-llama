"""Testing the DataClass."""

import unittest

from social_llama.data_processing.dataclass import DataClass


class TestDataClass(unittest.TestCase):
    """Test cases for the DataClass."""

    def setUp(self) -> None:
        """Setup the test case."""
        self.data_class = DataClass()
        self.labels = ["label1", "label2", "label3"]

    def test_label_to_idx_mapper(self) -> None:
        """Test the label_to_idx_mapper method."""
        self.data_class.set_labels(self.labels)
        expected_output = {"label1": 0, "label2": 1, "label3": 2}
        self.assertEqual(self.data_class.label_to_idx_mapper(), expected_output)

    def test_idx_to_label_mapper(self) -> None:
        """Test the idx_to_label_mapper method."""
        self.data_class.set_labels(self.labels)
        expected_output = {0: "label1", 1: "label2", 2: "label3"}
        self.assertEqual(self.data_class.idx_to_label_mapper(), expected_output)

    def test_set_labels(self) -> None:
        """Test the set_labels method."""
        self.data_class.set_labels(self.labels)
        self.assertEqual(self.data_class.labels, self.labels)

    def test_preprocess_not_implemented(self) -> None:
        """Test the preprocess method."""
        with self.assertRaises(NotImplementedError):
            self.data_class.preprocess()
