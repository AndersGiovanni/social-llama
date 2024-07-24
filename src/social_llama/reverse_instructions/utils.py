"""Utility functions for the reverse instructions module."""


def calculate_total_costs_from_nested(
    data_list, price_per_million_completion_tokens, price_per_million_prompt_tokens
):
    """Calculate the total costs of the generated data with separate prices for completion and prompt tokens."""
    total_completion_tokens = 0
    total_prompt_tokens = 0

    # Loop through each item in the data list
    for item in data_list:
        # Loop through each key in the item (e.g., 'train', 'test', 'validation')
        for key in item:
            # Loop through each entry under the key
            for entry in item[key]:
                # Extract and sum the token counts
                try:
                    total_completion_tokens += entry["metadata"]["usage"][
                        "completion_tokens"
                    ]
                    total_prompt_tokens += entry["metadata"]["usage"]["prompt_tokens"]
                except KeyError:
                    # If the token counts are not available, set them to 0
                    total_completion_tokens += 0
                    total_prompt_tokens += 0

    # Calculate the total price for completion and prompt tokens separately
    total_price_completion = (
        total_completion_tokens * price_per_million_completion_tokens
    ) / 1000000
    total_price_prompt = (
        total_prompt_tokens * price_per_million_prompt_tokens
    ) / 1000000

    # Sum of the prices for the total price
    total_price = total_price_completion + total_price_prompt

    return {
        "total_completion_tokens": total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_price_completion": total_price_completion,
        "total_price_prompt": total_price_prompt,
        "total_price": total_price,
    }


def estimate_total_costs_from_sample(
    data_list,
    price_per_million_completion_tokens,
    price_per_million_prompt_tokens,
    num_samples,
):
    """Estimate the total costs based on the number of samples."""
    total_completion_tokens_per_sample = 0
    total_prompt_tokens_per_sample = 0

    # Loop through each item in the data list
    for item in data_list:
        # Loop through each key in the item (e.g., 'train', 'test', 'validation')
        for key in item:
            # Loop through each entry under the key
            for entry in item[key]:
                # Extract and sum the token counts for one sample
                total_completion_tokens_per_sample += entry["metadata"]["usage"][
                    "completion_tokens"
                ]
                total_prompt_tokens_per_sample += entry["metadata"]["usage"][
                    "prompt_tokens"
                ]

    # Estimate total completion and prompt tokens for all samples
    estimated_total_completion_tokens = total_completion_tokens_per_sample * num_samples
    estimated_total_prompt_tokens = total_prompt_tokens_per_sample * num_samples

    # Calculate the estimated total price for each type of token
    estimated_total_price_completion = (
        estimated_total_completion_tokens * price_per_million_completion_tokens
    ) / 1000000
    estimated_total_price_prompt = (
        estimated_total_prompt_tokens * price_per_million_prompt_tokens
    ) / 1000000

    # Total estimated price
    estimated_total_price = (
        estimated_total_price_completion + estimated_total_price_prompt
    )

    return {
        "total_completion_tokens_per_sample": total_completion_tokens_per_sample,
        "total_prompt_tokens_per_sample": total_prompt_tokens_per_sample,
        "estimated_total_completion_tokens": estimated_total_completion_tokens,
        "estimated_total_prompt_tokens": estimated_total_prompt_tokens,
        "estimated_total_price_completion": estimated_total_price_completion,
        "estimated_total_price_prompt": estimated_total_price_prompt,
        "estimated_total_price": estimated_total_price,
    }
