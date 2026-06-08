from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPSILON = 1e-12


def normalize_counts(counts: Iterable[float]) -> np.ndarray:
    """Convert counts to a smoothed probability vector."""
    probs = np.asarray(list(counts), dtype=float)
    if probs.size == 0 or probs.sum() <= 0:
        return np.zeros_like(probs, dtype=float)
    probs = probs / probs.sum()
    probs = probs + EPSILON
    return probs / probs.sum()


def kl_divergence(model_probs: Iterable[float], human_probs: Iterable[float]) -> float:
    """KL(model || human)."""
    p = np.asarray(list(model_probs), dtype=float) + EPSILON
    q = np.asarray(list(human_probs), dtype=float) + EPSILON
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def normalized_entropy(probs: Iterable[float]) -> float:
    values = list(probs)
    p = np.asarray(values, dtype=float)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / math.log(len(values)))


def entropy_deviation(model_probs: Iterable[float], human_probs: Iterable[float]) -> float:
    """Absolute gap between model and human normalized entropy."""
    return abs(normalized_entropy(model_probs) - normalized_entropy(human_probs))


def metric_summary(model_probs: Iterable[float], human_probs: Iterable[float]) -> dict[str, float]:
    return {
        "KL divergence": kl_divergence(model_probs, human_probs),
        "ED": entropy_deviation(model_probs, human_probs),
        "Normalized entropy": normalized_entropy(model_probs),
    }
