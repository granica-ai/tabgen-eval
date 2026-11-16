"""Pack validated source observations without interpreting or repairing values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from tabgen_eval._options import ResolvedOptions
from tabgen_eval.execution.classifier.protocol import uses_classifier_protocol
from tabgen_eval.metrics.base import Metric
from tabgen_eval.preprocessing.arrow import try_prepare
from tabgen_eval.preprocessing.columns import PreparedColumns
from tabgen_eval.schema import Schema

from .dispatch import ExecutionSession, supports_prepared_evaluation
from .domains import SourceDomain, StorageKey, eligible_fixed_draw, prepare_metric_domains
from .sampling import FixedDraw, SamplingPlan


@dataclass(frozen=True)
class EvaluationSources:
    """Original comparison frames and optional compact encoded columns."""

    reference: pd.DataFrame
    generated: pd.DataFrame
    columns: PreparedColumns | None


def prepare_evaluation_sources(
    source: pd.DataFrame,
    generated: pd.DataFrame,
    schema: Schema,
    *,
    metrics: Sequence[Metric],
    retain_labels: bool = False,
) -> EvaluationSources:
    """Prepare validated inputs once, retaining all source observations."""
    options = {"retain_labels": True} if retain_labels else {}
    columns = (
        try_prepare(
            source,
            generated,
            schema,
            **options,
        )
        if metrics and supports_prepared_evaluation(metrics, schema)
        else None
    )
    return EvaluationSources(source, generated, columns)


@dataclass(frozen=True)
class PreparedMetricDomain:
    """A compatible group of computations and its shared source-storage session.

    Eligibility filters precede sampling. Compact column preparation only stores
    values and category identities; learned geometry, encoders, and models still
    fit on each computation's assigned draws or classifier training folds.
    """

    source: SourceDomain
    metrics: tuple[Metric, ...]
    sampling: FixedDraw | SamplingPlan
    session: ExecutionSession


def prepare_metric_sessions(
    metrics: Sequence[Metric],
    pools: Mapping[str, pd.DataFrame],
    schema: Schema,
    *,
    options: ResolvedOptions,
    n_replicates: int,
    details: bool = False,
) -> list[PreparedMetricDomain]:
    """Plan eligible pools, storage consumers, and sessions once for either mode.

    A generated pool requests paired comparisons. With test alone each draw
    uses two disjoint halves of that pool. Domains whose filtering leaves the
    same frame objects share one preparation/session, while their exclusion
    policies and row-usage records remain distinct.
    """
    domains = prepare_metric_domains(metrics, pools, schema)
    consumers: dict[StorageKey, list[Metric]] = {}
    for domain, members in domains:
        consumers.setdefault(domain.storage_key, []).extend(members)
    sessions: dict[StorageKey, ExecutionSession] = {}
    prepared: list[PreparedMetricDomain] = []
    for domain, members in domains:
        reference = domain.pools["test"]
        candidate = domain.pools.get("generated", reference)
        sampling = (
            eligible_fixed_draw(
                len(reference),
                len(candidate),
                seed=options["seed"],
                uncertainty=options["uncertainty"],
            )
            if "generated" in pools
            else SamplingPlan(len(reference), len(reference) // 2)
        )
        key = domain.storage_key
        if key not in sessions:
            shared = consumers[key]
            classifier_preprocessing = any(uses_classifier_protocol(metric) for metric in shared)
            retain_labels = classifier_preprocessing or (
                details and any(metric.name == "conditional_fidelity" for metric in shared)
            )
            sources = prepare_evaluation_sources(
                reference,
                candidate,
                schema,
                metrics=shared if sampling.n_res else (),
                retain_labels=retain_labels,
            )
            sessions[key] = ExecutionSession(
                reference,
                candidate,
                schema,
                n_jobs=options["n_jobs"],
                n_eff=sampling.n_res,
                n_bootstrap=n_replicates,
                columns=sources.columns,
                retain_labels=retain_labels,
                classifier_preprocessing=classifier_preprocessing,
            )
        prepared.append(PreparedMetricDomain(domain, tuple(members), sampling, sessions[key]))
    return prepared
