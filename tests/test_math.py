"""Test cases for the __main__ module."""

import pytest

from social_llama.math import SimpleMath


def test_add() -> None:
    """Test the add method of SimpleMath."""
    assert SimpleMath.add(1, 2) == 3
    assert SimpleMath.add(-1, 1) == 0
    assert SimpleMath.add(0, 0) == 0


def test_subtract() -> None:
    """Test the subtract method of SimpleMath."""
    assert SimpleMath.subtract(5, 3) == 2
    assert SimpleMath.subtract(2, 5) == -3
    assert SimpleMath.subtract(0, 0) == 0


def test_multiply() -> None:
    """Test the multiply method of SimpleMath."""
    assert SimpleMath.multiply(2, 3) == 6
    assert SimpleMath.multiply(-2, 3) == -6
    assert SimpleMath.multiply(2, 0) == 0


def test_divide() -> None:
    """Test the divide method of SimpleMath."""
    assert SimpleMath.divide(8, 2) == 4
    assert SimpleMath.divide(-8, 2) == -4

    # Testing division by zero using pytest.raises to check if it raises the correct exception
    with pytest.raises(ValueError, match="Division by zero is not allowed."):
        SimpleMath.divide(8, 0)
