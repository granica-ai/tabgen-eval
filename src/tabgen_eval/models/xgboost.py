"""Shared XGBoost model fitting and prediction for C2ST and supervised utility.

This module owns the CPU/CUDA engine, model lifecycle, and fit diagnostics.
Metric definitions, preprocessing, scoring, and resampling live in their own
modules. Importing this module defers importing the required XGBoost dependency until fitting.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Sequence
from numbers import Integral
from types import ModuleType
from typing import TYPE_CHECKING

from numpy.typing import ArrayLike
from sklearn.exceptions import NotFittedError

from tabgen_eval._typing import FloatArray, IntArray, TaskKind
from tabgen_eval.execution.xgboost.types import BoostingModel, FeatureOptions, NativeFeatures

from .errors import FitUnavailable

if TYPE_CHECKING:
    from torch import Tensor

    from tabgen_eval.execution.xgboost.types import ModelInput


class XGBoostModel:
    """Default CPU/CUDA classifier or regressor with reusable fitted state.

    ``fit`` returns this wrapper and retains the fitted XGBoost estimator in
    ``model`` and JSON-compatible training diagnostics in ``diagnostics``.
    ``predict`` reuses that model; ``fit_predict`` combines the two and returns
    ``(predictions, diagnostics)`` for metric adapters.

    There is no tuning or early stopping. ``fit_threads`` is a CPU thread budget,
    not a boosting-round limit. Missing or broken dependencies raise directly;
    engine fit/prediction failures raise ``FitUnavailable`` for metric-level
    flagging. Each fold or bootstrap fit should use its own wrapper instance.

    Devices are explicit: ``cpu`` (the default), ``cuda``, or ``cuda:<index>``.
    Automatic selection belongs to the metric execution layer, which has the
    workload and memory budgets needed to select a device before preprocessing.
    """

    def __init__(self, *, task: TaskKind = "classification", device: str = "cpu") -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        self._validate_device(device)
        self.task = task
        self.device = device
        self.model: BoostingModel | None = None
        self.diagnostics: dict[str, object] = {}
        self._engine: ModuleType | None = None
        self.feature_names: list[str] | None = None
        self.feature_types: list[str] | None = None

    @staticmethod
    def _validate_device(device: str) -> None:
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", device):
            raise ValueError(
                "XGBoostModel device must be 'cpu', 'cuda', or 'cuda:<index>'; "
                "automatic device selection belongs to the metric execution layer"
            )

    @staticmethod
    def execution_device(device: str) -> str:
        """Canonicalize an explicit engine device; reject unresolved requests.

        CPU and explicit ordinals never probe CUDA. An unindexed CUDA request
        follows the current Torch device when available, otherwise CUDA 0.
        """
        XGBoostModel._validate_device(device)
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            from tabgen_eval.execution.torch.tensors import torch_module

            torch = torch_module(required=False)
            index = (
                torch.cuda.current_device()
                if torch is not None and torch.cuda.is_available()
                else 0
            )
            return f"cuda:{index}"
        return device

    @staticmethod
    def load_backend() -> ModuleType:
        """Load/check the optional engine without hiding broken dependencies.

        This can be called before data checks so a missing installation remains
        distinct from a statistically noncomputable fit.
        """
        try:
            xgb = importlib.import_module("xgboost")
        except ModuleNotFoundError as error:
            if error.name != "xgboost":
                raise
            raise ModuleNotFoundError(
                "XGBoost is a required dependency. Reinstall 'tabgen-eval' or install 'xgboost>=2.1'.",
                name="xgboost",
            ) from error
        version = tuple(int(v) for v in xgb.__version__.split(".")[:2])
        if version < (2, 1):
            raise ImportError(
                "XGBoost metrics require XGBoost >=2.1; upgrade xgboost or xgboost-cpu"
            )
        return xgb

    def fit(
        self,
        train: ModelInput,
        labels: ArrayLike | Tensor,
        *,
        seed: int,
        fit_threads: int = 1,
        feature_types: Sequence[str] | None = None,
        feature_names: Sequence[str] | None = None,
        categorical_cardinalities: Sequence[int] | None = None,
    ) -> XGBoostModel:
        """Fit one default model on the selected device; return ``self``.

        Native categorical inputs supply ``feature_types`` (``q``/``c``), original
        ``feature_names``, and training vocabulary sizes for the categorical
        columns only. They force category-versus-rest splits, without external
        indicator expansion. Omitted metadata preserves the utility-model path.
        Completing the default boosting budget marks training as
        successful; it does not assert a convex optimality certificate.
        Refitting replaces the previous model; a failed refit leaves it unfitted.
        """
        if fit_threads < 1:
            raise ValueError("fit_threads must be positive")
        self.model = None
        self.diagnostics = {}
        self.feature_names = None
        self.feature_types = None
        metadata = self._feature_metadata(
            train, feature_types, feature_names, categorical_cardinalities
        )
        xgb = self.load_backend()
        self._engine = xgb
        if self.device.startswith("cuda") and not xgb.build_info().get("USE_CUDA", False):
            raise ImportError(
                "CUDA XGBoost requires the full CUDA-enabled 'xgboost>=2.1' package; "
                "replace xgboost-cpu with xgboost."
            )
        device = self.execution_device(self.device)
        model_cls = xgb.XGBClassifier if self.task == "classification" else xgb.XGBRegressor
        model_options: dict[str, object] = dict(
            device=device,
            tree_method="hist",
            n_jobs=fit_threads,
            random_state=int(seed % (2**32 - 1)),
        )
        if metadata:
            model_options.update(
                feature_types=metadata["feature_types"],
                enable_categorical=True,
                max_cat_to_onehot=max(metadata["categorical_cardinalities"], default=0) + 1,
            )
        model = model_cls(**model_options)
        try:
            from tabgen_eval.execution.xgboost.arrays import engine_array

            train_view = engine_array(train)
            # The sklearn facade validates class identities on the host. A label
            # vector is small; avoid CuPy's sorting/JIT machinery for this check.
            # Feature matrices remain on their selected device throughout.
            label_view = (
                labels.detach().cpu().numpy()
                if type(labels).__module__.startswith("torch")
                else labels
            )
            model.fit(train_view, label_view)
            booster = model.get_booster()
            if metadata:
                # Preserve arbitrary original names without imposing XGBoost's
                # restricted feature-name spelling on users' table schemas.
                booster.set_attr(tabgen_eval_preprocessing=json.dumps(metadata))
            rounds = int(booster.num_boosted_rounds())
            completed = rounds == model.get_num_boosting_rounds()
            configuration = json.loads(booster.save_config())
            # The seed is sufficient provenance; transient RNG state is not a setting.
            learner = configuration["learner"]
            actual_device = learner["generic_param"]["device"]
            if device.startswith("cuda") and actual_device != device:
                raise RuntimeError(
                    f"XGBoost requested {device!r} but selected {actual_device!r}; "
                    "check GPU availability and the requested device index."
                )
            learner["generic_param"].pop("rng_state", None)
            trees = int(learner["gradient_booster"]["gbtree_model_param"]["num_trees"])
            self.diagnostics = {
                "solver": "xgboost-hist",
                "device": actual_device,
                "converged": completed,
                "training_completed": completed,
                "iterations": rounds,
                "trees": trees,
                "fit_threads": int(fit_threads),
                "xgboost_version": xgb.__version__,
                "random_state": model.random_state,
                "model_config": configuration,
                "reason": "training_budget_completed"
                if completed
                else "training_budget_incomplete",
            }
            if metadata:
                self.diagnostics["categorical_policy"] = "one-category-versus-rest"
                self.diagnostics["input_features"] = metadata
                self.diagnostics["engine_feature_precision"] = "float32"
                self.feature_names = metadata["feature_names"]
                self.feature_types = metadata["feature_types"]
            self.model = model
        except xgb.core.XGBoostError as error:
            raise self._unavailable(error) from error
        return self

    @staticmethod
    def _feature_metadata(
        train: ModelInput,
        feature_types: Sequence[str] | None,
        feature_names: Sequence[str] | None,
        cardinalities: Sequence[int] | None,
    ) -> NativeFeatures | None:
        """Validate compact native-category descriptors before calling the engine."""
        if feature_types is None:
            if feature_names is not None or cardinalities is not None:
                raise ValueError("feature_names/cardinalities require feature_types")
            return None
        types = list(feature_types)
        if len(types) != train.shape[1] or any(t not in {"q", "c"} for t in types):
            raise ValueError("feature_types must contain one q/c entry per input column")
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"f{i}" for i in range(len(types))]
        )
        if (
            len(names) != len(types)
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("feature_names must uniquely name every input column")
        counts = [] if cardinalities is None else list(cardinalities)
        if len(counts) != types.count("c") or any(
            isinstance(k, bool) or not isinstance(k, Integral) or k < 1 for k in counts
        ):
            raise ValueError("categorical_cardinalities must give each training vocabulary size")
        counts = [int(k) for k in counts]
        # XGBoost categorical codes are represented as float32 integers. Reserve
        # the next exact code for held-out values absent from training.
        if any(k >= 2**24 for k in counts):
            raise ValueError("categorical vocabulary exceeds exact float32 code capacity")
        return {
            "feature_names": names,
            "feature_types": types,
            "categorical_cardinalities": counts,
            "unknown_category": "nonmatching-code",
        }

    def predict(self, test: ModelInput, *, prediction: str = "margin") -> FloatArray | IntArray:
        """Predict using the fitted model, without retraining or preprocessing.

        ``margin`` returns raw decision scores (C2ST), ``probability`` returns
        class probabilities (utility AUROC), and ``value`` returns ordinary
        predictions (utility regression). Probability requires classification.
        Calling this before a successful fit raises ``NotFittedError``.
        """
        if prediction not in {"margin", "probability", "value"}:
            raise ValueError("prediction must be 'margin', 'probability', or 'value'")
        if prediction == "probability" and self.task != "classification":
            raise ValueError("probability predictions require classification")
        if self.model is None:
            raise NotFittedError("Call XGBoostModel.fit before predict")
        assert self._engine is not None
        try:
            from tabgen_eval.execution.xgboost.arrays import engine_array

            test_view = engine_array(test)
            if self.feature_types is not None:
                import numpy as np

                matrix = self._engine.DMatrix(
                    test_view,
                    feature_types=self.feature_types,
                    enable_categorical=True,
                    nthread=self.diagnostics["fit_threads"],
                )
                scores = self.model.get_booster().predict(
                    matrix, output_margin=prediction == "margin"
                )
                if prediction == "probability":
                    return np.column_stack((1 - scores, scores)) if scores.ndim == 1 else scores
                if prediction == "value" and self.task == "classification":
                    return (
                        (scores > 0.5).astype(np.int64)
                        if scores.ndim == 1
                        else scores.argmax(axis=1)
                    )
                return scores
            if prediction == "probability":
                return self.model.predict_proba(test_view)
            return self.model.predict(test_view, output_margin=prediction == "margin")
        except self._engine.core.XGBoostError as error:
            raise self._unavailable(error) from error

    def fit_predict(
        self,
        train: ModelInput,
        labels: ArrayLike | Tensor,
        test: ModelInput,
        *,
        seed: int,
        fit_threads: int = 1,
        prediction: str = "margin",
        feature_types: Sequence[str] | None = None,
        feature_names: Sequence[str] | None = None,
        categorical_cardinalities: Sequence[int] | None = None,
    ) -> tuple[FloatArray | IntArray, dict[str, object]]:
        """Call ``fit`` then ``predict``; return predictions and fit diagnostics.

        The returned diagnostics are a copy so metric adapters can add fold or
        scoring metadata without modifying the wrapper's training diagnostics.
        """
        metadata: FeatureOptions = {}
        if (
            feature_types is not None
            or feature_names is not None
            or categorical_cardinalities is not None
        ):
            metadata = {
                "feature_types": feature_types,
                "feature_names": feature_names,
                "categorical_cardinalities": categorical_cardinalities,
            }
        self.fit(train, labels, seed=seed, fit_threads=fit_threads, **metadata)
        return self.predict(test, prediction=prediction), dict(self.diagnostics)

    def _unavailable(self, error: BaseException) -> FitUnavailable:
        """Translate engine failures into the shared fitted-metric failure contract."""
        assert self._engine is not None
        return FitUnavailable(
            "xgboost_fit_failed",
            solver="xgboost-hist",
            training_completed=False,
            xgboost_version=self._engine.__version__,
            error=str(error).splitlines()[0],
        )
