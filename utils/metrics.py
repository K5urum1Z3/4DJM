from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = labels == 1
    negative = labels == 0
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_metrics(labels: Iterable[int | float], probabilities: Iterable[float], threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(list(labels), dtype=np.int64)
    y_score = np.asarray(list(probabilities), dtype=np.float64)
    if y_true.size == 0 or y_true.shape != y_score.shape:
        raise ValueError("Labels and probabilities must be non-empty and have matching shape")
    y_pred = (y_score >= threshold).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    negative_precision = _safe_div(tn, tn + fn)
    negative_recall = specificity
    negative_f1 = _safe_div(2 * negative_precision * negative_recall, negative_precision + negative_recall)
    positive_f1 = _safe_div(2 * precision * recall, precision + recall)
    macro_f1 = (positive_f1 + negative_f1) / 2.0
    return {
        "count": int(y_true.size),
        "threshold": float(threshold),
        "accuracy": _safe_div(tp + tn, y_true.size),
        "precision": precision,
        "recall": recall,
        "f1": positive_f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "auroc": binary_auroc(y_true, y_score),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def grouped_metrics(records: list[dict[str, Any]], group_key: str, threshold: float = 0.5) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[group_key])].append(record)
    return {
        group: binary_metrics(
            [record["label"] for record in rows],
            [record["probability"] for record in rows],
            threshold,
        )
        for group, rows in sorted(groups.items())
    }
