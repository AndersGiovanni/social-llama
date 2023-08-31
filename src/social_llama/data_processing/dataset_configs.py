"""Configurations for the datasets used in the Social Llama project."""

from social_llama.config import DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED
from social_llama.config import DatasetConfig


SOCIAL_DIMENSIONS_CONFIG = DatasetConfig(
    name="social_dimensions",
    pretty_name="Social Dimensions",
    path=DATA_DIR_SOCIAL_DIMENSIONS_PROCESSED / "labeled_dataset.json",
    prompt_prefix="""The text is a social media post. The text conveys one or more social dimensions.
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
    """,
    prompt_template="Text: {text}\nSocial Dimensions: {response_good}",
    labels=[
        "social_support",
        "conflict",
        "trust",
        "fun",
        "similarity",
        "identity",
        "respect",
        "romance",
        "knowledge",
        "power",
        "other",
    ],
    max_generated_tokens=124,
    num_few_shot_examples=0,
)
