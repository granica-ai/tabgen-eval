"""Prediction agreement from reusable model outputs on identical observations."""

from __future__ import annotations

import numpy as np

from tabgen_eval._typing import FloatArray
from tabgen_eval.preprocessing.categories import CategoryVocabulary


def classification_agreement(
    generated: FloatArray | None,
    reference: FloatArray | None,
    generated_classes: CategoryVocabulary | None,
    reference_classes: CategoryVocabulary | None,
) -> float:
    """Fraction of equal predicted typed labels across independent vocabularies.

    Class indices have no shared meaning between independently fitted models.
    Decode each argmax through its vocabulary keys before comparing labels.
    """
    if generated is None or reference is None or not len(generated):
        return float("nan")
    assert generated_classes is not None and reference_classes is not None
    generated, reference = np.asarray(generated), np.asarray(reference)
    if generated.shape[0] != reference.shape[0]:
        raise ValueError("prediction agreement requires identical scoring rows")
    generated_labels = np.argmax(generated, axis=1)
    reference_labels = np.argmax(reference, axis=1)
    return float(
        np.mean(
            [
                generated_classes.keys[g] == reference_classes.keys[r]
                for g, r in zip(generated_labels, reference_labels, strict=True)
            ]
        )
    )


def regression_disagreement(generated: FloatArray | None, reference: FloatArray | None) -> float:
    """Mean squared difference between predictions, in squared target units."""
    if generated is None or reference is None or not len(generated):
        return float("nan")
    generated, reference = np.asarray(generated), np.asarray(reference)
    if generated.shape != reference.shape:
        raise ValueError("prediction agreement requires identical scoring rows")
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.mean((generated - reference) ** 2))
