"""Helper functions for evaluation."""

import collections
import re
import string
from typing import List
from typing import Tuple


def find_substring_indices(s: str, substrings: List[str]) -> List[int]:
    """Find all indices of a list of substrings within a string.

    Args:
        s (str): The string to search.
        substrings (List[str]): The substrings to search for.

    Returns:
        List[int]: The indices of the substrings within the string.
    """
    indices = []
    for substring in substrings:
        start = 0
        while start < len(s):
            start = s.find(substring, start)
            if start == -1:
                break
            indices.extend(range(start, start + len(substring)))
            start += len(substring)
    return sorted(set(indices))


def get_span_f1(predictions: List[int], gold: List[int]) -> float:
    """Compute F1 score based on Jaccard similarity.

    Args:
        predictions (List[int]): The predicted indices.
        gold (List[int]): The gold indices.

    Returns:
        float: The F1 score.
    """
    if not gold:
        return 1.0 if not predictions else 0.0
    nom = 2 * len(set(predictions) & set(gold))
    denom = len(set(predictions)) + len(set(gold))
    return nom / denom


def extract_spans(text: str) -> List[str]:
    """Extract spans based on quotes or the original string.

    Args:
        text (str): The text to extract spans from.

    Returns:
        List[str]: The extracted spans.
    """
    quoted = re.findall(r'"(.*?)"', text)
    return quoted if quoted else [text]


def longest_common_substring(s1: str, s2: str) -> str:
    """Find the longest common substring between two strings.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        str: The longest common substring.
    """
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0

    for x in range(1, len(s1) + 1):
        for y in range(1, len(s2) + 1):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0

    return s1[x_longest - longest : x_longest].strip()


def normalize_answer(s: str) -> str:
    """Normalize text: lower text and remove punctuation, articles, and extra whitespace.

    Args:
        s (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def get_tokens(s: str) -> List[str]:
    """Tokenize the text.

    Args:
        s (str): The text to tokenize.

    Returns:
        List[str]: The tokenized text.
    """
    return normalize_answer(s).split() if s else []


def compute_exact(a_gold: str, a_pred: str) -> int:
    """Compute exact match between two strings.

    Args:
        a_gold (str): The gold answer.
        a_pred (str): The predicted answer.

    Returns:
        int: 1 if the answers match exactly, 0 otherwise.

    """
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_metrics(a_gold: str, a_pred: str) -> Tuple[float, float, float]:
    """Compute F1, Precision, and Recall.

    Args:
        a_gold (str): The gold answer.
        a_pred (str): The predicted answer.

    Returns:
        Tuple[float, float, float]: The F1, Precision, and Recall scores.
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if not gold_toks or not pred_toks:
        return (
            int(gold_toks == pred_toks),
            int(gold_toks == pred_toks),
            int(gold_toks == pred_toks),
        )

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def get_mean(li: List[float]) -> float:
    """Compute the mean of a list of numbers.

    Args:
        li (List[float]): The list of numbers.

    Returns:
        float: The mean of the numbers.
    """
    return sum(li) / len(li) if li else 0.0


def get_all_f1(groundtruth: List[str], answer: List[str]) -> Tuple[float, float, float]:
    """Compute mean F1, Precision, and Recall.

    Args:
        groundtruth (List[str]): The list of gold answers.
        answer (List[str]): The list of predicted answers.

    Returns:
        Tuple[float, float, float]: The mean F1, Precision, and Recall scores.
    """
    f1s = [compute_metrics(g, a)[0] for g, a in zip(groundtruth, answer)]
    ps = [compute_metrics(g, a)[1] for g, a in zip(groundtruth, answer)]
    rs = [compute_metrics(g, a)[2] for g, a in zip(groundtruth, answer)]

    return get_mean(ps), get_mean(rs), get_mean(f1s)


def label_check(prediction: str, labels: List[str]) -> str:
    """Check if the prediction contains a label."""
    # sort by length of label to avoid matching substrings
    for label in sorted(labels, key=len, reverse=True):
        # Check if we can find the true label in the prediction
        pattern = rf"\b{label}\b"
        if re.search(pattern, prediction, re.IGNORECASE):
            return label

    return "FALSE"


def label_finder(prediction, labels):
    """Find the first occurrence of a label in a string."""
    earliest_pos = len(prediction)
    earliest_string = None

    for label in labels:
        pattern = rf"\b{re.escape(label)}\b"
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            earliest_string = label

    return earliest_string if earliest_string else "FALSE"
