"""Module for basic mathematical operations.

This module contains a class for performing basic arithmetic
operations such as addition, subtraction, multiplication, and division.
"""

from typing import Union


class SimpleMath:
    """Class to perform basic mathematical operations."""

    @staticmethod
    def add(
        number_1: Union[float, int], number_2: Union[float, int]
    ) -> Union[float, int]:
        """Return the sum of two numbers.

        Args:
            number_1 (Union[float, int]): The first number.
            number_2 (Union[float, int]): The second number.

        Returns:
            Union[float, int]: Sum of number_1 and number_2.
        """
        return number_1 + number_2

    @staticmethod
    def subtract(
        number_1: Union[float, int], number_2: Union[float, int]
    ) -> Union[float, int]:
        """Return the difference between two numbers.

        Args:
            number_1 (Union[float, int]): The first number.
            number_2 (Union[float, int]): The second number.

        Returns:
            Union[float, int]: Difference between number_1 and number_2.
        """
        return number_1 - number_2

    @staticmethod
    def multiply(
        number_1: Union[float, int], number_2: Union[float, int]
    ) -> Union[float, int]:
        """Return the product of two numbers.

        Args:
            number_1 (Union[float, int]): The first number.
            number_2 (Union[float, int]): The second number.

        Returns:
            Union[float, int]: Product of number_1 and number_2.
        """
        return number_1 * number_2

    @staticmethod
    def divide(number_1: Union[float, int], number_2: Union[float, int]) -> float:
        """Return the division of two numbers.

        Args:
            number_1 (Union[float, int]): Numerator.
            number_2 (Union[float, int]): Denominator.

        Returns:
            float: Result of division of number_1 by number_2.

        Raises:
            ValueError: If number_2 is 0 (to avoid division by zero).
        """
        if number_2 == 0:
            raise ValueError("Division by zero is not allowed.")
        return number_1 / number_2
