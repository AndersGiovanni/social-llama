"""Defining configuration classes for the reverse instructions module."""


class ReverseInstructionsPrompts:
    """Configuration class for the reverse instructions prompts."""

    # Path to the reverse instructions prompts
    def reverse_instruction_cls(self) -> tuple[str, str]:
        """Return the prompt for the reverse instruction task."""
        return (
            """You are a helpful assistant helping in creating instructions for a text classification task.""",
            """Instruction: X
Input: {text}
Labels: {label_list}
Output: {label}
What kind of instruction could "Output" be the answer to given "Input" and "Labels"? Please make only an instruction for the task and include brief descriptions of the labels.
Begin your answer with 'X: '""",
        )
