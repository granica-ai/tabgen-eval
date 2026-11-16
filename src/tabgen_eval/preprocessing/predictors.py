"""Training-only native categorical preprocessing for supervised XGBoost.

Predictive MLE and model agreement share this fitted state. Host ingestion
decodes pandas labels; the selected compiled CPU/CUDA backend fits numeric and
category state and assembles compact, unscaled model inputs on that device.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabgen_eval._typing import FloatArray, Precision
from tabgen_eval.models import FitUnavailable
from tabgen_eval.schema import Schema

from .categories import CategoryVocabulary, category_key
from .contracts import POLICY_VERSION, CategoryState, NumericState
from .sources import ObservationPrecision, observation_precision, prepare_sources

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

    from tabgen_eval.execution.torch.tensors import TensorTable


def match_predictor_precision(
    frames: Iterable[pd.DataFrame], schema: Schema, dtype: Precision = "float64"
) -> tuple[tuple[pd.DataFrame, ...], ObservationPrecision]:
    """Match observation precision across roles, without fitting on test values.

    The common width depends only on declared numeric input dtypes. This uses
    the same compiled source quantizer as C2ST and never changes caller frames.
    It runs before fitting the supervised predictors.
    """
    import torch

    from .columns import numeric_values

    frames = tuple(frames)
    names = tuple(schema.numeric)
    precision = observation_precision(frames[0], frames[0], names, dtype)
    bits = dict(precision["observation_bits"])
    for frame in frames[1:]:
        other = observation_precision(frames[0], frame, names, dtype)
        bits = {name: min(bits[name], other["observation_bits"][name]) for name in names}
    precision.update(
        observation_bits=bits,
        input_dtypes={name: [str(frame[name].dtype) for frame in frames] for name in names},
    )
    result, cache = [], {}
    for frame in frames:
        if id(frame) in cache:
            result.append(cache[id(frame)])
            continue
        own = observation_precision(frame, frame, names, dtype)["observation_bits"]
        changed = [name for name in names if own[name] > bits[name]]
        # A requested FP32 execution also rounds an originally FP64 source.
        changed.extend(
            name
            for name in names
            if name not in changed
            and dtype == "float32"
            and str(frame[name].dtype) not in {"float16", "float32"}
        )
        if not changed:
            result.append(frame)
            cache[id(frame)] = frame
            continue
        from tabgen_eval.execution.torch.cpu.extension import library

        values = torch.from_numpy(np.column_stack([numeric_values(frame[c]) for c in changed]))
        widths = torch.tensor([bits[c] for c in changed], dtype=torch.int64)
        quantized = library().preprocessing_quantize(values, widths, getattr(torch, dtype)).numpy()
        output = frame.copy(deep=False)
        for index, name in enumerate(changed):
            output[name] = quantized[:, index]
        result.append(output)
        cache[id(frame)] = output
    return tuple(result), precision


@dataclass(frozen=True)
class PredictorPreprocessor:
    """Reusable training-only state; one output column per categorical predictor."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    vocabularies: tuple[CategoryVocabulary, ...]
    numeric_state: NumericState
    category_state: CategoryState
    feature_names: tuple[str, ...]
    feature_types: tuple[str, ...]
    categorical_cardinalities: tuple[int, ...]
    device: str
    dtype: Precision
    memory_limit: float
    diagnostics: dict[str, object]

    @property
    def width(self) -> int:
        return len(self.feature_types)

    @staticmethod
    def _budget(rows: int, numeric: int, categorical: int, limit: float) -> int:
        # Sources, outputs, masks and worst-case training vocabulary state are
        # bounded by N*P, never N*sum(category cardinalities).
        estimate = rows * (64 * numeric + 112 * categorical + 32) + 65536
        if estimate > limit:
            raise FitUnavailable(
                "preprocessing_or_memory_limit", estimated_bytes=estimate, memory_limit_bytes=limit
            )
        return estimate

    @classmethod
    def _fit(
        cls,
        training: pd.DataFrame,
        schema: Schema,
        memory_limit: float,
        *,
        device: str = "cpu",
        dtype: Precision = "float64",
    ) -> tuple[PredictorPreprocessor, TensorTable]:
        import torch

        from tabgen_eval.schema import Schema

        numeric = tuple(c for c in schema.numeric if c != schema.target)
        categorical = tuple(c for c in schema.categorical if c != schema.target)
        estimated = cls._budget(len(training), len(numeric), len(categorical), memory_limit)
        features = Schema(numeric=list(numeric), categorical=list(categorical))
        sources = prepare_sources(training, training, features, device=device, dtype=dtype)
        table = sources.reference
        backend = import_module(f"tabgen_eval.preprocessing.{table.numeric.device.type}")
        # Shared classifier primitives accept two training pools. Aliasing the
        # one supervised training pool fits exactly its observed support;
        # presence, retained columns and the fixed unscaled tree policy agree.
        ns = NumericState(*backend.fit_numeric(table.numeric, table.numeric, False))
        cards = torch.tensor(sources.cardinalities, dtype=torch.int64, device=table.numeric.device)
        cs = CategoryState(*backend.fit_categories(table.categorical, table.categorical, cards))
        keep = ns.keep.cpu().tolist()
        if not keep and not categorical:
            raise FitUnavailable("no_usable_predictor_features")
        names = [numeric[i] for i in keep] + list(categorical)
        # XGBoost's persisted feature descriptors require unique string names.
        feature_names = (
            tuple(names)
            if all(isinstance(n, str) for n in names)
            else tuple(f"f{i}" for i in range(len(names)))
        )
        sizes = tuple((cs.feature_offsets[1:] - cs.feature_offsets[:-1]).cpu().tolist())
        vocabularies = tuple(
            CategoryVocabulary(tuple(category_key(v) for v in values), values)
            for values in sources.category_values
        )
        state = cls(
            numeric,
            categorical,
            vocabularies,
            ns,
            cs,
            feature_names,
            ("q",) * len(keep) + ("c",) * len(categorical),
            sizes,
            str(table.numeric.device),
            dtype,
            memory_limit,
            dict(
                policy_version=POLICY_VERSION,
                policy="xgboost",
                device=str(table.numeric.device),
                numerical_scaling="none",
                categorical_encoding="native_one_vs_rest",
                categorical_scaling="none",
                target_excluded=True,
                estimated_bytes=estimated,
                source_precision=sources.precision,
                dropped_numeric_columns=[str(n) for i, n in enumerate(numeric) if i not in keep],
            ),
        )
        return state, table

    @classmethod
    def fit(
        cls,
        training: pd.DataFrame,
        schema: Schema,
        memory_limit: float,
        *,
        device: str = "cpu",
        dtype: Precision = "float64",
    ) -> PredictorPreprocessor:
        state, _ = cls._fit(training, schema, memory_limit, device=device, dtype=dtype)
        return state

    @classmethod
    def fit_transform(
        cls,
        training: pd.DataFrame,
        schema: Schema,
        memory_limit: float,
        *,
        device: str = "cpu",
        dtype: Precision = "float64",
    ) -> tuple[PredictorPreprocessor, Tensor]:
        """Fit and assemble training inputs with one source-ingestion pass."""
        state, table = cls._fit(training, schema, memory_limit, device=device, dtype=dtype)
        return state, state._assemble(table.numeric, table.categorical)

    def _assemble(self, numbers: Tensor, codes: Tensor, *, unseen: Tensor | None = None) -> Tensor:
        import torch

        backend = import_module(f"tabgen_eval.preprocessing.{self.device.split(':')[0]}")
        ns, cs = self.numeric_state, self.category_state
        values = backend.apply_numeric(
            numbers, ns.keep, ns.indicators, ns.impute, ns.center, ns.scale, False
        )
        mapped = backend.map_categories(
            codes, cs.mapping, cs.domain_offsets, cs.feature_offsets, False
        )
        if unseen is not None:
            sizes = cs.feature_offsets[1:] - cs.feature_offsets[:-1]
            mapped = torch.where(unseen, sizes[None, :], mapped)
        return backend.assemble(values, mapped, len(cs.active), False)

    def transform(self, frame: pd.DataFrame) -> Tensor:
        """Apply frozen state; unseen labels use a nonmatching categorical code."""
        import torch

        from tabgen_eval.schema import Schema

        self._budget(len(frame), len(self.numeric), len(self.categorical), self.memory_limit)
        # Shared ingestion/precision handling for numerics; labels are encoded
        # against the frozen training dictionaries, never a scoring vocabulary.
        sources = prepare_sources(
            frame, frame, Schema(numeric=list(self.numeric)), device=self.device, dtype=self.dtype
        )
        if self.categorical:
            host = np.column_stack(
                [
                    v.transform(frame[c])
                    for c, v in zip(self.categorical, self.vocabularies, strict=True)
                ]
            )
            codes = torch.as_tensor(host, dtype=torch.int64, device=self.device).contiguous()
        else:
            codes = torch.empty((len(frame), 0), dtype=torch.int64, device=self.device)
        unseen = codes == -2
        return self._assemble(sources.reference.numeric, codes.clamp_min(-1), unseen=unseen)


def predictor_design(
    training: pd.DataFrame, test: pd.DataFrame, schema: Schema, memory_limit: float
) -> tuple[FloatArray, FloatArray]:
    """Test adapter exposing native predictor designs for independent comparisons."""
    state, train = PredictorPreprocessor.fit_transform(training, schema, memory_limit)
    return train.numpy(), state.transform(test).numpy()
