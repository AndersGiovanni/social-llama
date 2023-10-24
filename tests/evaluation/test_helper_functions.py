"""Test the evaluation helper functions."""
import unittest

from social_llama.evaluation.helper_functions import (
    compute_exact,  # Replace with the actual import
)
from social_llama.evaluation.helper_functions import compute_metrics
from social_llama.evaluation.helper_functions import extract_spans
from social_llama.evaluation.helper_functions import find_substring_indices
from social_llama.evaluation.helper_functions import get_all_f1
from social_llama.evaluation.helper_functions import get_mean
from social_llama.evaluation.helper_functions import get_span_f1
from social_llama.evaluation.helper_functions import get_tokens
from social_llama.evaluation.helper_functions import longest_common_substring
from social_llama.evaluation.helper_functions import normalize_answer


class TestEvaluationHelper(unittest.TestCase):
    """Test the evaluation helper functions."""

    def test_find_substring_indices(self):
        """Test the find_substring_indices function."""
        self.assertEqual(
            find_substring_indices("hello world", ["ll", "wo"]), [2, 3, 6, 7]
        )

    def test_get_span_f1(self):
        """Test the get_span_f1 function."""
        self.assertAlmostEqual(
            get_span_f1([0, 1, 4, 5], [0, 1, 6]), 0.5714285714285714, places=7
        )

    def test_extract_spans(self):
        """Test the extract_spans function."""
        self.assertEqual(extract_spans('"hello" "world"'), ["hello", "world"])
        self.assertEqual(extract_spans("no quotes here"), ["no quotes here"])

    def test_longest_common_substring(self):
        """Test the longest_common_substring function."""
        self.assertEqual(longest_common_substring("hello", "world"), "l")

    def test_normalize_answer(self):
        """Test the normalize_answer function."""
        self.assertEqual(normalize_answer("The quick, brown fox."), "quick brown fox")

    def test_get_tokens(self):
        """Test the get_tokens function."""
        self.assertEqual(get_tokens("The quick, brown fox."), ["quick", "brown", "fox"])

    def test_compute_exact(self):
        """Test the compute_exact function."""
        self.assertEqual(compute_exact("The quick, brown fox.", "quick brown fox"), 1)

    def test_compute_metrics(self):
        """Test the compute_metrics function."""
        f1, p, r = compute_metrics("The quick, brown fox.", "quick brown fox")
        self.assertAlmostEqual(f1, 1.0, places=7)
        self.assertAlmostEqual(p, 1.0, places=7)
        self.assertAlmostEqual(r, 1.0, places=7)

    def test_get_mean(self):
        """Test the get_mean function."""
        self.assertAlmostEqual(get_mean([1.0, 2.0, 3.0]), 2.0, places=7)

    def test_get_all_f1(self):
        """Test the get_all_f1 function."""
        mean_p, mean_r, mean_f1 = get_all_f1(
            ["The quick, brown fox.", "hello"], ["quick brown fox", "hi"]
        )
        self.assertNotEqual(mean_f1, 1.0)
        self.assertNotEqual(mean_p, 1.0)
        self.assertNotEqual(mean_r, 1.0)


if __name__ == "__main__":
    unittest.main()
