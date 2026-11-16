"""Python classifier folds release their storage before allocating another fold."""

import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from tabgen_eval.execution.python.classifier import compute_prepared
from tabgen_eval.models import FitUnavailable
from tabgen_eval.preprocessing.contracts import PreparedFold


@pytest.mark.parametrize("failure", [None, "fit", "preparation"])
def test_python_fold_storage_is_released_before_next_preparation(monkeypatch, torch, failure):
    import tabgen_eval.preprocessing as preprocessing

    references = []
    prepared_folds, fitted_folds = [], []

    def prepare(*args, **kwargs):
        # Check before constructing any new fold storage, including after a
        # failed fit or an earlier partially prepared fold.
        assert all(reference() is None for reference in references)
        fold = args[4]
        prepared_folds.append(fold)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        data = PreparedFold(
            torch.zeros((4, 1)),
            labels[:, None].clone(),
            labels.clone(),
            labels.clone(),
            None,
            None,
            ("x",),
            ("q",),
            (),
            {},
        )
        references.append(weakref.ref(data))
        if failure == "preparation" and fold == 1:
            raise FitUnavailable("test_preparation_failure")
        return data

    def fit(train, labels, test, **kwargs):
        fold = kwargs["seed"] - 42
        fitted_folds.append(fold)
        references.extend(weakref.ref(value) for value in (train, labels, test))
        # Retaining scores can also retain the entire scoring matrix.
        scores = test[:, 0]
        references.append(weakref.ref(scores))
        if failure == "fit" and fold == 1:
            raise FitUnavailable("test_fit_failure")
        return scores, {"converged": True}

    monkeypatch.setattr(preprocessing, "fit_fold", prepare)
    sources = SimpleNamespace(
        cardinalities=(),
        numeric_names=("x",),
        categorical_names=(),
        category_values=(),
        precision={},
    )
    result = compute_prepared(
        None,
        None,
        None,
        None,
        sources=sources,
        seed=42,
        point=True,
        fit_threads=1,
        cv_folds=3,
        memory_limit_mb=1,
        fit_predict=fit,
    )
    assert prepared_folds == [0, 1, 2]
    assert fitted_folds == ([0, 2] if failure == "preparation" else [0, 1, 2])
    assert len(result.folds) == 3
    assert all(reference() is None for reference in references)
    if failure:
        assert np.isnan(result.value)
        assert result.folds[1]["reason"] == f"test_{failure}_failure"
        assert result.folds[2]["converged"]
    else:
        assert result.value == 1.0 and result.converged
