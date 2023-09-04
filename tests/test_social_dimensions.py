"""Testing the social dimensions dataclass."""


import unittest

from social_llama.data_processing.social_dimensions import Sample
from social_llama.data_processing.social_dimensions import SocialDimensions


class TestSocialDimensions(unittest.TestCase):
    """Test cases for the SocialDimensions."""

    def setUp(self):
        """Set up the test cases."""
        self.social_dimensions_zero_shot = SocialDimensions(task="zero-shot")
        self.social_dimensions_few_shot = SocialDimensions(task="zero-shot")
        self.social_dimensions_cot = SocialDimensions(task="cot")
        self.sample_data = {
            "text": "example text",
            "h_text": "example h_text",
            "response_good": "good",
            "response_bad": "bad",
        }

    def test_sample_class(self):
        """Test the Sample class."""
        sample = Sample(idx="1", **self.sample_data)
        self.assertEqual(sample.idx, "1")
        self.assertEqual(sample.text, "example text")

    def test_prompt_function(self):
        """Test the prompt function."""
        data = Sample(
            idx="0",
            text=""""Fried rice" in the US (at least the Northeast US, where I am from) typically refers to the non-plain rice ... Please note also that most of the food we Americans refer to as "Chinese Food" is American in that one ... As such, I might assume that your rice in question might be American Chinese food style fried rice. What\'s in that "American" rice served in Malaysia, besides rice?""",  # noqa
            h_text="As such, I might assume that your rice in question might be American Chinese food style fried rice.",
            response_good="power",
            response_bad="similarity",
        )
        prompt_zero_shot = self.social_dimensions_zero_shot._prompt_function(data)
        prompt_cot = self.social_dimensions_cot._prompt_function(data)

        self.assertEqual(
            prompt_zero_shot,
            """Text: "Fried rice" in the US (at least the Northeast US, where I am from) typically refers to the non-plain rice ... Please note also that most of the food we Americans refer to as "Chinese Food" is American in that one ... As such, I might assume that your rice in question might be American Chinese food style fried rice. What\'s in that "American" rice served in Malaysia, besides rice?\nSocial Dimensions: power""",  # noqa
        )

        self.assertEqual(
            prompt_cot,
            """Text: "Fried rice" in the US (at least the Northeast US, where I am from) typically refers to the non-plain rice ... Please note also that most of the food we Americans refer to as "Chinese Food" is American in that one ... As such, I might assume that your rice in question might be American Chinese food style fried rice. What\'s in that "American" rice served in Malaysia, besides rice?\nThe text exhibits power over the behavior and outcomes of another. In particular in the part \'As such, I might assume that your rice in question might be American Chinese food style fried rice.\'.\n    The social dimensions are: power""",  # noqa
        )


if __name__ == "__main__":
    unittest.main()
