"""Typed interfaces at the XGBoost and optional device-array boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypedDict

import numpy as np
from numpy.typing import ArrayLike

from tabgen_eval._typing import FloatArray, IntArray

if TYPE_CHECKING:
    from torch import Tensor

    ModelInput: TypeAlias = FloatArray | IntArray | Tensor


class NativeFeatures(TypedDict):
    """Training-fitted native categorical descriptors persisted with a model."""

    feature_names: list[str]
    feature_types: list[str]
    categorical_cardinalities: list[int]
    unknown_category: str


class FeatureOptions(TypedDict, total=False):
    """Optional native-category keyword arguments passed to a model fit."""

    feature_types: Sequence[str] | None
    feature_names: Sequence[str] | None
    categorical_cardinalities: Sequence[int] | None


class CudaArray(Protocol):
    """Device-array metadata consumed synchronously by the optional engine."""

    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]

    @property
    def __cuda_array_interface__(self) -> dict[str, object]: ...


class Booster(Protocol):
    """Native model methods used to record training provenance and predict."""

    def set_attr(self, **attributes: str) -> None: ...
    def num_boosted_rounds(self) -> int: ...
    def save_config(self) -> str: ...
    def predict(self, data: object, *, output_margin: bool = False) -> FloatArray: ...


class BoostingModel(Protocol):
    """Supported subset of the engine's fitted classifier/regressor interface."""

    random_state: int

    def fit(self, train: object, labels: ArrayLike) -> BoostingModel: ...
    def get_booster(self) -> Booster: ...
    def get_num_boosting_rounds(self) -> int: ...
    def predict(self, test: object, *, output_margin: bool = False) -> FloatArray | IntArray: ...
    def predict_proba(self, test: object) -> FloatArray: ...
