"""Sequence-level metrics used by both the accuracy bench and the load test."""


from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance over characters, two rows of memory."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def character_error_rate(prediction: str, reference: str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return edit_distance(prediction, reference) / len(reference)


def accuracy(prediction: str, reference: str) -> float:
    """The challenge's per-image score: max(0, 1 - CER)."""
    return max(0.0, 1.0 - character_error_rate(prediction, reference))
