"""Automatic headroom, source accounting and shared task allowances."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabgen_eval.execution.runtime import memory
from tabgen_eval.metrics import MLE, LinearC2ST, NonLinearC2ST, Shape

pytestmark = pytest.mark.unit
GIB = 1024**3


def test_default_allowance_tracks_capacity_without_two_gib_ceiling(monkeypatch):
    metric = LinearC2ST()
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 20 * GIB)
    plan = memory.ResourcePlanner().resolve(metric, "cpu")
    assert plan.aggregate_workspace_bytes == 16 * GIB
    assert metric.memory_limit_mb == 16 * 1024
    assert metric.gpu_workspace_mb == 16 * 1024
    assert json.loads(json.dumps(plan.to_dict())) == plan.to_dict()


def test_source_preflight_and_postplacement_do_not_double_charge(monkeypatch):
    metric = Shape()
    free = [10 * GIB]
    monkeypatch.setattr(memory, "host_available_bytes", lambda: free[0])
    before = memory.ResourcePlanner().resolve(metric, "cpu", new_source_bytes=2 * GIB)
    free[0] -= 2 * GIB
    after = memory.ResourcePlanner().resolve(metric, "cpu", persistent_input_bytes=2 * GIB)
    assert before.task_workspace_bytes == after.task_workspace_bytes == 8 * GIB * 4 // 5
    assert after.new_source_bytes == 0
    assert after.persistent_input_bytes == 2 * GIB


def test_concurrent_tasks_reduce_to_feasible_count_before_division(monkeypatch):
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 5 * GIB)
    metric = LinearC2ST()
    plan = memory.ResourcePlanner().resolve(
        metric, "cpu", concurrent_tasks=8, minimum_task_bytes=2 * GIB
    )
    assert plan.concurrent_tasks == 2
    assert plan.task_workspace_bytes == 2 * GIB
    assert metric.memory_limit_mb == 2048


def test_wide_compact_solver_estimate_includes_rank_state(monkeypatch):
    metric = LinearC2ST(saga_rank=64)
    rows, cardinality = 200_000, 100_000
    required = memory.classifier_minimum_bytes(metric, rows, 0, 1, (cardinality,))
    width = cardinality + 3
    assert required >= 20 * width * metric.saga_rank * 8
    context = SimpleNamespace(
        n_numeric=0,
        n_categorical=1,
        inputs=SimpleNamespace(cardinalities=(cardinality,)),
        execution_info={"device": "cpu", "persistent_input_bytes": rows * 8},
    )
    monkeypatch.setattr(memory, "host_available_bytes", lambda: required * 5 // 2)
    plan = memory.plan_classifier_memory(metric, context, rows, concurrent_tasks=8)
    assert plan.concurrent_tasks <= 2
    assert plan.task_workspace_bytes >= required


def test_private_explicit_constructor_budgets_remain_fixed(monkeypatch):
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 100 * GIB)
    metric = LinearC2ST(gpu_workspace_mb=17, memory_limit_mb=31)
    memory.ResourcePlanner().resolve(metric, "cpu")
    assert metric.gpu_workspace_mb == 17
    assert metric.memory_limit_mb == 31


def test_cached_classifier_sources_are_not_reserved_again(monkeypatch):
    import pandas as pd

    from tabgen_eval import Schema
    from tabgen_eval.execution.runtime.dispatch import ExecutionSession

    monkeypatch.setattr(memory, "host_available_bytes", lambda: 8 * GIB)
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    session = ExecutionSession(frame, frame, Schema(numeric=["x"]))
    linear = session.context_for(LinearC2ST(device="cpu"))
    workload = session._workload(NonLinearC2ST(device="cpu"))
    monkeypatch.setattr(
        session,
        "_workload",
        lambda metric: SimpleNamespace(**vars(workload), persistent_bytes=lambda dtype: 12 * GIB),
    )
    forest = session.context_for(NonLinearC2ST(device="cpu"))
    assert forest.reference is linear.reference


def test_exhausted_memory_does_not_create_an_artificial_minimum(monkeypatch):
    monkeypatch.setattr(memory, "host_available_bytes", lambda: memory.MIB)
    with pytest.raises(MemoryError, match="Insufficient available memory"):
        memory.ResourcePlanner().resolve(Shape(), "cpu")


def test_torch_cache_is_not_credited_to_xgboost(monkeypatch):
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(memory, "cuda_available_bytes", lambda device: (20 * GIB, 8 * GIB, 4 * GIB))
    torch = memory.ResourcePlanner().resolve(Shape(), "cuda:2")
    forest = memory.ResourcePlanner().resolve(NonLinearC2ST(), "cuda:2")
    assert torch.task_workspace_bytes == 24 * GIB * 4 // 5
    assert forest.task_workspace_bytes == 16 * GIB
    assert forest.cached_torch_bytes == 8 * GIB
    assert forest.reusable_torch_cache_bytes == 0


def test_cuda_staging_respects_host_headroom(monkeypatch):
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 2 * GIB)
    monkeypatch.setattr(memory, "cuda_available_bytes", lambda device: (64 * GIB, 0, 0))
    with pytest.raises(MemoryError, match="host memory to stage"):
        memory.ResourcePlanner().resolve(Shape(), "cuda:1", new_source_bytes=3 * GIB)


def test_cached_blocks_awaiting_other_streams_are_not_counted_as_reusable(monkeypatch):
    from tabgen_eval.execution.torch import tensors

    cuda = SimpleNamespace(
        mem_get_info=lambda device: (20 * GIB, 80 * GIB),
        memory_reserved=lambda device: 10 * GIB,
        memory_allocated=lambda device: 2 * GIB,
        memory_stats=lambda device: {
            "active_bytes.all.current": 5 * GIB,
            "inactive_split_bytes.all.current": GIB,
        },
    )
    torch = SimpleNamespace(cuda=cuda, device=lambda value: value)
    monkeypatch.setattr(tensors, "torch_module", lambda **kwargs: torch)
    assert memory.cuda_available_bytes("cuda:3") == (20 * GIB, 8 * GIB, 4 * GIB)


def test_multiclass_forest_estimates_include_per_class_work():
    binary = memory.xgboost_workspace_bytes(10_000, 4, (10,), output_groups=1)
    multiclass = memory.xgboost_workspace_bytes(10_000, 4, (10,), output_groups=100)
    assert multiclass > 10 * binary
    assert memory.xgboost_model_bytes(100) > memory.xgboost_model_bytes(1)


def test_worker_lease_subtracts_its_resident_design_before_engine_fit(monkeypatch):
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 64 * GIB)
    original = MLE()
    worker = memory.leased_metric(original, 4 * GIB)
    plan = memory.ResourcePlanner().resolve(worker, "cpu", owned_task_bytes=2 * GIB)
    assert worker is not original
    assert plan.task_workspace_bytes == 2 * GIB
    assert not hasattr(original, "_memory_lease_bytes")


def test_process_worker_overhead_is_reserved_before_workspace_division(monkeypatch):
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 10 * GIB)
    plan = memory.ResourcePlanner().resolve(
        MLE(), "cpu", concurrent_tasks=8, minimum_task_bytes=GIB, worker_overhead_bytes=GIB
    )
    assert plan.concurrent_tasks == 4
    assert plan.worker_overhead_bytes == GIB
    assert plan.task_workspace_bytes == GIB
    assert (
        plan.concurrent_tasks * (plan.task_workspace_bytes + plan.worker_overhead_bytes) == 8 * GIB
    )


def test_serial_execution_does_not_charge_parent_process_again(monkeypatch):
    monkeypatch.setattr(memory, "host_available_bytes", lambda: 5 * GIB)
    plan = memory.ResourcePlanner().resolve(
        MLE(), "cpu", concurrent_tasks=8, minimum_task_bytes=3 * GIB, worker_overhead_bytes=2 * GIB
    )
    assert plan.concurrent_tasks == 1
    assert plan.worker_overhead_bytes == 0
    assert plan.task_workspace_bytes == 4 * GIB


def test_portable_available_ram_is_clamped_by_container_limit(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=64 * GIB))
    monkeypatch.setattr(memory, "_cgroup_available_bytes", lambda: 3 * GIB)
    assert memory.host_available_bytes() == 3 * GIB


@pytest.mark.parametrize("version", [1, 2])
def test_tightest_cgroup_ancestor_limits_headroom(tmp_path, monkeypatch, version):
    mount = tmp_path / "controller"
    child = mount / "parent" / "child"
    child.mkdir(parents=True)
    maximum = "memory.max" if version == 2 else "memory.limit_in_bytes"
    current = "memory.current" if version == 2 else "memory.usage_in_bytes"
    (mount / maximum).write_text(str(10 * GIB))
    (mount / current).write_text(str(2 * GIB))
    (child.parent / maximum).write_text(str(5 * GIB))
    (child.parent / current).write_text(str(4 * GIB))
    (child / maximum).write_text("max" if version == 2 else str(100 * GIB))
    (child / current).write_text(str(GIB))
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "cgroup").write_text(
        "0::/parent/child\n" if version == 2 else "7:memory:/parent/child\n"
    )
    (proc / "mountinfo").write_text(
        f"42 0 0:42 / {mount} rw - "
        + ("cgroup2 cgroup rw\n" if version == 2 else "cgroup cgroup rw,memory\n")
    )

    def paths(value):
        return (
            proc / Path(value).name
            if value in {"/proc/self/cgroup", "/proc/self/mountinfo"}
            else Path(value)
        )

    monkeypatch.setattr(memory, "Path", paths)
    assert memory._cgroup_available_bytes() == GIB
