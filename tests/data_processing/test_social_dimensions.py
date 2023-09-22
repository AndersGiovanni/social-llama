"""Testing the social dimensions dataclass."""


import unittest
from dataclasses import asdict
from typing import Dict

from datasets import Dataset

from social_llama.data_processing.social_dimensions import Sample
from social_llama.data_processing.social_dimensions import SocialDimensions


class TestSocialDimensions(unittest.TestCase):
    """Test cases for the SocialDimensions."""

    def setUp(self):
        """Set up the test cases."""
        self.social_dimensions_zero_shot = SocialDimensions(task="zero-shot")
        self.social_dimensions_few_shot = SocialDimensions(task="few-shot")
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

        print(prompt_zero_shot)
        print(prompt_cot)

        self.assertEqual(
            prompt_zero_shot,
            """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

You have the following knowledge about social dimensions of human interactions.
The social dimensions are 'social_support', 'conflict', 'trust', 'fun', 'similarity', 'identity',
'respect', 'romance', 'knowledge', 'power', and 'other'.
The social dimensions are defined as follows:
'social_support': Giving emotional or practical aid and companionship.
'conflict': Contrast or diverging views.
'trust': Will of relying on the actions or judgments of another.
'fun': Experiencing leisure, laughter, and joy.
'similarity': Shared interests, motivations or outlooks.
'identity': Shared sense of belonging to the same community or group.
'respect': Conferring status, respect, appreciation, gratitude, or admiration upon another.
'romance': Intimacy among people with a sentimental or sexual relationship.
'knowledge': Exchange of ideas or information; learning, teaching.
'power': Having power over the behavior and outcomes of another.
'other': If none of the above social dimensions apply.


<</SYS>>

Question: What is the social dimension of the following text?

Text: "Fried rice" in the US (at least the Northeast US, where I am from) typically refers to the non-plain rice ... Please note also that most of the food we Americans refer to as "Chinese Food" is American in that one ... As such, I might assume that your rice in question might be American Chinese food style fried rice. What's in that "American" rice served in Malaysia, besides rice?

Answer: power [/INST]
""",  # noqa
        )

        self.assertEqual(
            prompt_cot,
            """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

You have the following knowledge about social dimensions of human interactions.
The social dimensions are 'social_support', 'conflict', 'trust', 'fun', 'similarity', 'identity',
'respect', 'romance', 'knowledge', 'power', and 'other'.
The social dimensions are defined as follows:
'social_support': Giving emotional or practical aid and companionship.
'conflict': Contrast or diverging views.
'trust': Will of relying on the actions or judgments of another.
'fun': Experiencing leisure, laughter, and joy.
'similarity': Shared interests, motivations or outlooks.
'identity': Shared sense of belonging to the same community or group.
'respect': Conferring status, respect, appreciation, gratitude, or admiration upon another.
'romance': Intimacy among people with a sentimental or sexual relationship.
'knowledge': Exchange of ideas or information; learning, teaching.
'power': Having power over the behavior and outcomes of another.
'other': If none of the above social dimensions apply.


<</SYS>>

Text: "Fried rice" in the US (at least the Northeast US, where I am from) typically refers to the non-plain rice ... Please note also that most of the food we Americans refer to as "Chinese Food" is American in that one ... As such, I might assume that your rice in question might be American Chinese food style fried rice. What's in that "American" rice served in Malaysia, besides rice?
The text exhibits power over the behavior and outcomes of another. In particular in the part 'As such, I might assume that your rice in question might be American Chinese food style fried rice.'.
    Answer: power [/INST]
""",  # noqa
        )

    def test_convert_to_q_and_a(self):
        """Test the convert_to_q_and_a function."""
        # Test zero-shot
        data = Sample(
            idx="0",
            text=""""Fried rice" in the US (at least the Northeast US, where I am from) typically refers to the non-plain rice ... Please note also that most of the food we Americans refer to as "Chinese Food" is American in that one ... As such, I might assume that your rice in question might be American Chinese food style fried rice. What\'s in that "American" rice served in Malaysia, besides rice?""",  # noqa
            h_text="As such, I might assume that your rice in question might be American Chinese food style fried rice.",
            response_good="power",
            response_bad="similarity",
        )

        datasetdict = Dataset.from_list([asdict(data)])
        original_columns = datasetdict.column_names
        res: Dict = datasetdict.map(
            self.social_dimensions_zero_shot._convert_to_q_and_a,
            remove_columns=original_columns,
        )[0]

        self.assertEqual(
            res,
            {
                "prompt": "Question: The following text is a social media post. The text conveys one or more social dimensions, and your job is to determine the dimension. The social dimensions are 'social_support', 'conflict', 'trust', 'fun', 'similarity', 'identity', 'respect', 'romance', 'knowledge', 'power', and 'other'. \n\nText: \"Fried rice\" in the US (at least the Northeast US, where I am from) typically refers to the non-plain rice ... Please note also that most of the food we Americans refer to as \"Chinese Food\" is American in that one ... As such, I might assume that your rice in question might be American Chinese food style fried rice. What's in that \"American\" rice served in Malaysia, besides rice?\n\nAsnwer: ",  # noqa
                "chosen": "power",
                "rejected": "similarity",
            },
        )

        # Test few-shot
        data_few_shot = {
            "idx": [[25, 7, 10], [22, 20, 4]],
            "text": [
                [
                    "It is a a fucking weird time, when science and statistics are no longer relevant. Alright I'm with you. If you are not heterosexual then something did not develop properly in the brain. Lost me.",
                    '"No. Say it. I want you to say a homeless person would be the best choice."',
                    '"We are American English. Lower your shields and surrender your words. We will add your verbal and grammatical distinctiveness to our own. Your culture will adapt to service us. Resistance is futile."',
                ],
                [
                    "I would honestly prefer those who disagree to bring a real conversation to the table. People have tried bringing a real conversation to the table, and when they do, feminists try to shut it down. It makes sense to oppose the people who oppose the changes you're trying to make. The idea that radical feminism is just some fringe group with no real power is bullshit. This has been going on for a long time.",
                    "I get that he may not want to judge his friend Why? Reacting to your friends behavior (negative or positive) is normal and how friendships are built in the first place. We want to surround ourself with people we like after all.",
                    '"I\'m proud of you for being who you are." Let him know that he can confide in and trust you. Don\'t say you\'ve known - it dismisses the whole coming out process which is often a huge step and difficult ... if you say "i love you anyway" it suggests, "...despite this thing wrong with you". Just act as if nothing is different.',
                ],
            ],
            "h_text": [
                [
                    "If you are not heterosexual then something did not develop properly in the brain.",
                    'I want you to say a homeless person would be the best choice."',
                    "We will add your verbal and grammatical distinctiveness to our own.",
                ],
                [
                    "It makes sense to oppose the people who oppose the changes you're trying to make.",
                    "Reacting to your friends behavior (negative or positive) is normal and how friendships are built in the first place.",
                    "Let him know that he can confide in and trust you.",
                ],
            ],
            "response_good": [
                ["conflict", "social_support", "trust"],
                ["conflict", "social_support", "trust"],
            ],
            "response_bad": [
                ["trust", "similarity", "social_support"],
                ["knowledge", "fun", "respect"],
            ],
        }

        datasetdict_few_shot = Dataset.from_dict(data_few_shot)
        original_columns = datasetdict_few_shot.column_names
        res_few_shot = datasetdict_few_shot.map(
            self.social_dimensions_few_shot._convert_to_q_and_a,
            remove_columns=original_columns,
        )[0]

        self.assertEqual(
            res_few_shot,
            {
                "prompt": 'Text: It is a a fucking weird time, when science and statistics are no longer relevant. Alright I\'m with you. If you are not heterosexual then something did not develop properly in the brain. Lost me.\n\nAnswer: conflict\nText: "No. Say it. I want you to say a homeless person would be the best choice."\n\nAnswer: social_support\nText: "We are American English. Lower your shields and surrender your words. We will add your verbal and grammatical distinctiveness to our own. Your culture will adapt to service us. Resistance is futile."\n\nAnswer: ',
                "chosen": "trust",
                "rejected": "social_support",
            },
        )

        # Test few-shot CoT
        data_cot = {
            "idx": [[25, 7, 10], [22, 20, 4]],
            "text": [
                [
                    "It is a a fucking weird time, when science and statistics are no longer relevant. Alright I'm with you. If you are not heterosexual then something did not develop properly in the brain. Lost me.",
                    '"No. Say it. I want you to say a homeless person would be the best choice."',
                    '"We are American English. Lower your shields and surrender your words. We will add your verbal and grammatical distinctiveness to our own. Your culture will adapt to service us. Resistance is futile."',
                ],
                [
                    "I would honestly prefer those who disagree to bring a real conversation to the table. People have tried bringing a real conversation to the table, and when they do, feminists try to shut it down. It makes sense to oppose the people who oppose the changes you're trying to make. The idea that radical feminism is just some fringe group with no real power is bullshit. This has been going on for a long time.",
                    "I get that he may not want to judge his friend Why? Reacting to your friends behavior (negative or positive) is normal and how friendships are built in the first place. We want to surround ourself with people we like after all.",
                    '"I\'m proud of you for being who you are." Let him know that he can confide in and trust you. Don\'t say you\'ve known - it dismisses the whole coming out process which is often a huge step and difficult ... if you say "i love you anyway" it suggests, "...despite this thing wrong with you". Just act as if nothing is different.',
                ],
            ],
            "h_text": [
                [
                    "If you are not heterosexual then something did not develop properly in the brain.",
                    'I want you to say a homeless person would be the best choice."',
                    "We will add your verbal and grammatical distinctiveness to our own.",
                ],
                [
                    "It makes sense to oppose the people who oppose the changes you're trying to make.",
                    "Reacting to your friends behavior (negative or positive) is normal and how friendships are built in the first place.",
                    "Let him know that he can confide in and trust you.",
                ],
            ],
            "response_good": [
                ["conflict", "social_support", "trust"],
                ["conflict", "social_support", "trust"],
            ],
            "response_bad": [
                ["trust", "similarity", "social_support"],
                ["knowledge", "fun", "respect"],
            ],
        }

        datasetdict_cot = Dataset.from_dict(data_cot)
        original_columns = datasetdict_cot.column_names
        res_cot = datasetdict_cot.map(
            self.social_dimensions_cot._convert_to_q_and_a,
            remove_columns=original_columns,
        )[0]

        self.assertEqual(
            res_cot,
            {
                "prompt": "Text: It is a a fucking weird time, when science and statistics are no longer relevant. Alright I'm with you. If you are not heterosexual then something did not develop properly in the brain. Lost me.\nThe text exhibits contrast or diverging views. In particular in the part ''If you are not heterosexual then something did not develop properly in the brain.''.\nAnswer: conflict\nText: \"No. Say it. I want you to say a homeless person would be the best choice.\"\nThe text exhibits emotional or practical aid and companionship. In particular in the part ''I want you to say a homeless person would be the best choice.\"''.\nAnswer: social_support\nText: \"We are American English. Lower your shields and surrender your words. We will add your verbal and grammatical distinctiveness to our own. Your culture will adapt to service us. Resistance is futile.\"\n",
                "chosen": "The text exhibits will of relying on the actions or judgments of another. In particular in the part ''We will add your verbal and grammatical distinctiveness to our own.''.\nAnswer: trust\n",
                "rejected": "The text exhibits emotional or practical aid and companionship. In particular in the part ''We will add your verbal and grammatical distinctiveness to our own.''.\nAnswer: social_support\n",
            },
        )


if __name__ == "__main__":
    unittest.main()
