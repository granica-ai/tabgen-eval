"""cuBLAS/cuSOLVER, full-one-hot operators and GPU classifier protocol oracles."""

from __future__ import annotations

import ctypes
import gc

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import expit

from tabgen_eval import Schema
from tabgen_eval.execution.classifier.protocol import source_folds
from tabgen_eval.execution.runtime.dispatch import context_for
from tabgen_eval.metrics import LinearC2ST
from tests.support.evaluation import paired_report as evaluate
from tests.support.linear_c2st import compute_partitioned as linear_reference
from tests.support.metrics import configured_metric
from tests.support.options import make_options

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


def frames(dtype="float64"):
    rng = np.random.default_rng(1281)
    r = pd.DataFrame(
        {
            "x": rng.normal(size=240).astype(dtype),
            "y": rng.normal(size=240).astype(dtype),
            "c": rng.choice(["a", "b", None], 240),
        }
    )
    s = pd.DataFrame(
        {
            "x": (rng.normal(size=210) + 0.5).astype(dtype),
            "y": rng.normal(size=210).astype(dtype),
            "c": rng.choice(["a", "new", None], 210),
        }
    )
    r.loc[::23, "x"] = np.nan
    s.loc[::31, "y"] = np.nan
    return r, s, Schema(numeric=["x", "y"], categorical=["c"])


def test_cublas_and_cusolver_match_independent_matrix_oracles(torch, device, dtype, ops):
    rng = np.random.default_rng(1282)
    values = rng.normal(size=(137, 13)).astype(dtype)
    weights = rng.normal(size=13).astype(dtype)
    vector = torch.tensor(weights, device=device)
    for column_major in [False, True]:
        x = torch.tensor(values, device=device)
        if column_major:
            x = x.T.contiguous().T
        np.testing.assert_allclose(
            ops.mv(x, vector, False).cpu(),
            values.astype(float) @ weights,
            rtol=2e-6 if dtype == "float32" else 1e-13,
            atol=2e-6 if dtype == "float32" else 1e-13,
        )
        v = torch.ones(len(x), device=device, dtype=x.dtype)
        np.testing.assert_allclose(
            ops.mv(x, v, True).cpu(),
            values.astype(float).sum(0),
            rtol=3e-6 if dtype == "float32" else 1e-13,
            atol=3e-6 if dtype == "float32" else 1e-13,
        )
        matrix = ops.lower_gram(x)
        matrix.diagonal().add_(0.1)
        lower, info = ops.cholesky(matrix)
        solution, status = ops.solve(lower, vector)
        assert info.item() == status.item() == 0
        exact = np.linalg.solve(
            values.astype(float).T @ values.astype(float) + 0.1 * np.eye(13), weights
        )
        np.testing.assert_allclose(
            solution.cpu(),
            exact,
            rtol=3e-6 if dtype == "float32" else 1e-13,
            atol=3e-6 if dtype == "float32" else 1e-13,
        )
    invalid = torch.full((3, 3), float("nan"), device=device, dtype=getattr(torch, dtype)).T
    _, info = ops.cholesky(invalid)
    assert info.item() != 0


def test_fused_logistic_extreme_logits_have_selected_precision(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.triton.classifier import logistic

    z = torch.tensor(
        [-1000.0, -80.0, -2.0, 0.0, 2.0, 80.0, 1000.0], dtype=getattr(torch, dtype), device=device
    )
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=z.dtype, device=device)
    residual, curvature, loss = logistic(z, y)
    zn, yn = z.cpu().numpy().astype(float), y.cpu().numpy()
    expected = np.where(yn == 0, expit(zn), -expit(-zn)) / len(z)
    e = np.exp(-np.abs(zn))
    np.testing.assert_allclose(
        residual.cpu(),
        expected,
        rtol=2e-6 if dtype == "float32" else 1e-14,
        atol=1e-8 if dtype == "float32" else 1e-16,
    )
    np.testing.assert_allclose(
        curvature.cpu(),
        e / ((1 + e) ** 2 * len(z)),
        rtol=2e-6 if dtype == "float32" else 1e-14,
        atol=1e-8 if dtype == "float32" else 1e-16,
    )
    np.testing.assert_allclose(
        loss.cpu(),
        np.logaddexp(0, np.where(yn == 1, -zn, zn)),
        rtol=2e-6 if dtype == "float32" else 1e-14,
        atol=1e-8 if dtype == "float32" else 1e-16,
    )
    assert residual.dtype == curvature.dtype == loss.dtype == z.dtype


def test_triton_loss_difference_retains_small_armijo_changes(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.triton.classifier import loss_difference

    logits = np.array([-80.0, -2.0, 0.0, 2.0, 80.0], dtype=dtype)
    changes = np.full(len(logits), 1e-7, dtype=dtype)
    labels = np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=dtype)
    result = loss_difference(
        torch.tensor(logits, device=device),
        torch.tensor(changes, device=device),
        torch.tensor(labels, device=device),
    )
    z = (1 - 2 * labels.astype(np.longdouble)) * logits.astype(np.longdouble)
    delta = (1 - 2 * labels.astype(np.longdouble)) * changes.astype(np.longdouble)
    expected = np.logaddexp(0, z + delta) - np.logaddexp(0, z)
    np.testing.assert_allclose(
        result.cpu(),
        expected.astype(float),
        rtol=2e-6 if dtype == "float32" else 2e-11,
        atol=1e-40,
    )
    assert result.dtype == getattr(torch, dtype)


def test_native_newton_loss_difference_retains_small_armijo_changes(torch, device, dtype):
    from tabgen_eval.execution.torch.cuda.classifier.device import operators
    from tests.support.cuda import launch_kernel

    logits = np.array([-80.0, -2.0, 0.0, 2.0, 80.0], dtype=dtype)
    changes = np.full(len(logits), 1e-7, dtype=dtype)
    labels = np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=dtype)
    inputs = [torch.tensor(values, device=device) for values in (logits, changes, labels)]
    blocks = (len(logits) + 255) // 256
    result = torch.empty(blocks, dtype=torch.float64, device=device)
    _, ptx = operators(inputs[0], algorithm="newton")
    with torch.cuda.device(device):
        launch_kernel(
            ptx,
            "newton_loss_difference",
            blocks=blocks,
            threads=256,
            arguments=[
                *(ctypes.c_void_p(tensor.data_ptr()) for tensor in inputs),
                ctypes.c_void_p(result.data_ptr()),
                ctypes.c_int(len(logits)),
            ],
            stream=torch.cuda.current_stream(device).cuda_stream,
        )
    # Check the production Newton kernel's block reductions. Subtracting total
    # FP32 losses would discard this change; extended precision is independent
    # of both the native stable-difference formula and its reduction.
    z = (1 - 2 * labels.astype(np.longdouble)) * logits.astype(np.longdouble)
    delta = (1 - 2 * labels.astype(np.longdouble)) * changes.astype(np.longdouble)
    expected = (np.logaddexp(0, z + delta) - np.logaddexp(0, z)).mean()
    assert float(result.sum().item()) == pytest.approx(
        float(expected), rel=2e-6 if dtype == "float32" else 2e-11, abs=1e-40
    )


def test_default_ridge_fp32_wide_direct_fit_converges(torch, device):
    rng_r, rng_s = np.random.default_rng(810), np.random.default_rng(811)
    numeric = [f"n{i}" for i in range(50)]
    categorical = [f"c{i}" for i in range(50)]
    tables = []
    for rng in [rng_r, rng_s]:
        table = pd.DataFrame(rng.normal(size=(10000, 50)), columns=numeric)
        for column in categorical:
            table[column] = rng.integers(10, size=len(table)).astype(str)
        tables.append(table)
    schema = Schema(numeric=numeric, categorical=categorical)
    r, s = tables
    rf, sf = source_folds(5000, 5000, 3, np.random.default_rng(81))
    metric = LinearC2ST(device=device, dtype="float32", solver="direct")
    outcome = context_for(metric, r, s, schema).compute_classifier(
        metric,
        np.arange(5000)[None],
        np.arange(5000)[None],
        rf[None],
        sf[None],
        [81],
        point=True,
    )[0]
    assert outcome.converged, outcome.folds
    assert all(fold["features"] == 550 for fold in outcome.folds)
    assert all(fold["gradient_squared"] <= metric.tolerance**2 for fold in outcome.folds)
    assert all(np.isfinite(fold["factorization_damping"]) for fold in outcome.folds)
    expected = linear_reference(metric, r.iloc[:5000], s.iloc[:5000], schema, rf, sf, seed=81)
    assert expected.converged
    assert outcome.value == pytest.approx(expected.value, abs=2e-5)


def test_training_fold_preprocessing_and_implicit_adjoint_match_explicit_design(
    torch, device, dtype
):
    from tabgen_eval.execution.torch.cuda.classifier.data import prepare
    from tabgen_eval.execution.torch.cuda.classifier.solver import derivatives, hessian_vector

    r, s, schema = frames(dtype)
    r["dropped"] = np.nan
    s["dropped"] = np.nan
    schema.numeric.append("dropped")
    rf, sf = source_folds(len(r), len(s), 3, np.random.default_rng(1283))
    metric = LinearC2ST(device=device, dtype=dtype, ridge=0.03)
    context = context_for(metric, r, s, schema)
    af, bf = (torch.tensor(v, device=device) for v in [rf, sf])
    fold = prepare(
        context.reference,
        context.candidate,
        af,
        bf,
        0,
        context.inputs.cardinalities,
        metric.native_options(point=True),
        128 * 1024**2,
        0,
    )
    # Independently fit the agreed training-only transforms in NumPy. Category
    # ordering follows prepared typed identities, not a model implementation.
    training = np.concatenate(
        (r.loc[rf != 0, schema.numeric], s.loc[sf != 0, schema.numeric])
    ).astype(dtype)
    scoring = np.concatenate(
        (r.loc[rf == 0, schema.numeric], s.loc[sf == 0, schema.numeric])
    ).astype(dtype)
    real_numeric = r.loc[rf != 0, schema.numeric].to_numpy(dtype=dtype)
    synthetic_numeric = s.loc[sf != 0, schema.numeric].to_numpy(dtype=dtype)
    finite = np.isfinite(training)
    keep = np.flatnonzero(finite.any(0))
    indicators = np.flatnonzero(finite.any(0) & ~finite.all(0))
    impute = []
    for column in keep:
        observed = real_numeric[np.isfinite(real_numeric[:, column]), column]
        if not len(observed):
            observed = synthetic_numeric[np.isfinite(synthetic_numeric[:, column]), column]
        impute.append(observed.astype(np.float64).mean())
    impute = np.array(impute, dtype=dtype)
    fitted = np.where(finite[:, keep], training[:, keep], impute)
    center = fitted.astype(np.float64).mean(0).astype(dtype)
    scale = fitted.astype(np.float64).std(0).astype(dtype)
    scale[scale == 0] = 1

    def numeric(values):
        raw = values[:, keep]
        return np.column_stack(
            (
                (np.where(np.isfinite(raw), raw, impute) - center) / scale,
                ~np.isfinite(values[:, indicators]),
            )
        ).astype(dtype)

    exact_train, exact_test = numeric(training), numeric(scoring)
    from tabgen_eval.preprocessing.categories import factorize_categories

    for column in schema.categorical:
        (reference_codes, synthetic_codes), _ = factorize_categories(r[column], s[column])
        train_codes = np.concatenate((reference_codes[rf != 0], synthetic_codes[sf != 0]))
        test_codes = np.concatenate((reference_codes[rf == 0], synthetic_codes[sf == 0]))
        active = np.unique(train_codes)
        exact_train = np.column_stack((exact_train, train_codes[:, None] == active))
        exact_test = np.column_stack((exact_test, test_codes[:, None] == active))
    labels = np.concatenate((np.zeros((rf != 0).sum()), np.ones((sf != 0).sum())))
    # The independent oracle excludes an intercept; GPU tensors include it explicitly.
    np.testing.assert_allclose(
        fold.train.dense[:, :-1].cpu(),
        exact_train,
        rtol=2e-6 if dtype == "float32" else 1e-13,
        atol=2e-6 if dtype == "float32" else 1e-13,
    )
    np.testing.assert_allclose(
        fold.test.dense[:, :-1].cpu(),
        exact_test,
        rtol=2e-6 if dtype == "float32" else 1e-13,
        atol=2e-6 if dtype == "float32" else 1e-13,
    )
    rng = np.random.default_rng(1284)
    w = torch.tensor(rng.normal(size=fold.train.d), dtype=getattr(torch, dtype), device=device)
    v = torch.tensor(rng.normal(size=fold.train.n), dtype=w.dtype, device=device)
    expected = fold.train.dense @ w
    expected_adjoint = fold.train.dense.T @ v
    fold.train.dense = None
    fold.train.invert()
    torch.testing.assert_close(
        fold.train.mv(w),
        expected,
        rtol=2e-6 if dtype == "float32" else 1e-13,
        atol=2e-6 if dtype == "float32" else 1e-13,
    )
    torch.testing.assert_close(
        fold.train.transpose_mv(v),
        expected_adjoint,
        rtol=3e-6 if dtype == "float32" else 1e-13,
        atol=3e-6 if dtype == "float32" else 1e-13,
    )
    x = np.column_stack((exact_train, np.ones(len(exact_train))))
    wn = w.cpu().numpy().astype(float)
    g, h = derivatives(fold.train, fold.labels, w, 0.03)
    gradient = x.T @ ((expit(x @ wn) - labels) / len(labels))
    gradient[:-1] += 0.03 * wn[:-1]
    np.testing.assert_allclose(
        g.cpu(), gradient, atol=2e-6 if dtype == "float32" else 1e-13, rtol=0
    )
    hv = hessian_vector(fold.train, h, w, 0.03)
    probability = expit(x @ wn)
    exact_hv = x.T @ ((probability * (1 - probability) / len(labels)) * (x @ wn))
    exact_hv[:-1] += 0.03 * wn[:-1]
    np.testing.assert_allclose(
        hv.cpu(), exact_hv, atol=3e-6 if dtype == "float32" else 1e-13, rtol=0
    )


def test_gpu_optimizer_matches_independent_optimum(torch, device, dtype):
    solver = "direct"
    from tabgen_eval.execution.torch.cuda.classifier.data import Design, Fold
    from tabgen_eval.execution.torch.cuda.classifier.solver import fit

    rng = np.random.default_rng(1285)
    x = np.column_stack((rng.normal(size=(137, 4)), np.ones(137))).astype(dtype)
    y = rng.binomial(1, expit(x @ np.array([0.7, -0.2, 0.4, 0.1, -0.1]))).astype(dtype)
    matrix = torch.tensor(x, device=device).T.contiguous().T
    labels = torch.tensor(y, device=device)
    design = Design(
        matrix[:, :-1].contiguous(),
        torch.empty((len(x), 0), dtype=torch.int64, device=device),
        matrix.new_empty(0),
        matrix.new_empty(0),
        dense=matrix,
    )
    fold = Fold(
        design,
        design,
        labels,
        labels,
        solver,
        "none",
        64 * 1024**2,
    )
    metric = LinearC2ST(ridge=0.03, tolerance=1e-8 if dtype == "float64" else 1e-5)
    w, state = fit(fold, metric.native_options(), 81)
    assert state["converged"], state
    xn, yn = x.astype(float), y.astype(float)

    def objective(weights):
        z = xn @ weights
        return np.logaddexp(0, np.where(yn == 1, -z, z)).mean() + 0.015 * (weights[:-1] ** 2).sum()

    optimum = minimize(objective, np.zeros(x.shape[1]), method="BFGS", options={"gtol": 1e-10})
    np.testing.assert_allclose(
        w.cpu(), optimum.x, rtol=0, atol=2e-4 if dtype == "float32" else 3e-6
    )


def test_complete_fold_results_and_tied_auc(torch, device, dtype):
    solver = "direct"
    from tabgen_eval.execution.torch.cuda.classifier.solver import auc

    r, s, schema = frames(dtype)
    rf, sf = source_folds(len(r), len(s), 3, np.random.default_rng(1286))
    metric = LinearC2ST(
        device=device,
        dtype=dtype,
        solver=solver,
        ridge=0.03,
        tolerance=1e-6 if dtype == "float64" else 1e-4,
    )
    context = context_for(metric, r, s, schema)
    outcome = context.compute_classifier(
        metric,
        np.arange(len(r))[None],
        np.arange(len(s))[None],
        rf[None],
        sf[None],
        [2**64 - 2],
        point=True,
    )[0]
    expected = linear_reference(metric, r, s, schema, rf, sf, seed=8, point=True)
    assert outcome.converged and expected.converged
    assert outcome.value == pytest.approx(expected.value, abs=2e-4)
    assert {v["solver"] for v in outcome.folds} == {solver}
    scores = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=getattr(torch, dtype))
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device, dtype=scores.dtype)
    assert auc(scores, labels) == 0.5


def test_wide_design_uses_bounded_implicit_full_one_hot(torch, device, dtype):
    rng = np.random.default_rng(1287)
    schema = Schema(numeric=["x"], categorical=["a", "b"])
    r = pd.DataFrame(
        {
            "x": rng.normal(size=360).astype(dtype),
            "a": np.arange(360).astype(str),
            "b": np.arange(360)[::-1].astype(str),
        }
    )
    s = r.copy()
    rf, sf = source_folds(len(r), len(s), 3, np.random.default_rng(1288))
    metric = LinearC2ST(
        device=device,
        dtype=dtype,
        solver="sketchysaga",
        saga_rank=8,
        # Native preparation charges the queried ~6.6 MB cuSOLVER scratch.
        memory_limit_mb=8,
        ridge=0.03,
    )
    context = context_for(metric, r, s, schema)
    outcome = context.compute_classifier(
        metric,
        np.arange(len(r))[None],
        np.arange(len(s))[None],
        rf[None],
        sf[None],
        [81],
        point=True,
    )[0]
    assert outcome.converged, outcome.folds
    assert all(v["solver"] == "sketchysaga" for v in outcome.folds)
    assert all(v["features"] >= 400 for v in outcome.folds)
    expected = linear_reference(metric, r, s, schema, rf, sf, seed=81, point=True)
    # The 8 MiB budget fits this FP32 dense reference but not its FP64
    # counterpart. Both GPU cases must retain all features through compact
    # storage; the reference must either score chance or decline explicitly.
    assert outcome.value == pytest.approx(0.5, abs=2e-4)
    if dtype == "float32":
        assert expected.value == pytest.approx(0.5, abs=2e-4)
    else:
        assert not np.isfinite(expected.value)
        assert all(fold["reason"] == "preprocessing_memory_limit" for fold in expected.folds)


@pytest.mark.parametrize("bootstrap", [0, 2])
def test_report_provenance_failure_flags_and_precision_settings(torch, device, bootstrap):
    r, s, schema = frames()
    before = (
        torch.cuda.current_device(),
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            options = make_options(
                device=device,
                uncertainty="both" if bootstrap else "none",
                n_replicates=max(1, bootstrap),
            )
            report = evaluate(
                r,
                s,
                schema,
                metrics=[
                    configured_metric(
                        LinearC2ST,
                        options,
                        ridge=0.03,
                        tolerance=1e-06,
                    )
                ],
                options=options,
            )
        options = make_options(
            uncertainty="both" if bootstrap else "none",
            n_replicates=max(1, bootstrap),
        )
        expected = evaluate(
            r,
            s,
            schema,
            metrics=[
                configured_metric(
                    LinearC2ST,
                    options,
                    ridge=0.03,
                    tolerance=1e-06,
                )
            ],
            options=options,
        )
        actual, other = report.results[0].stats["auc"], expected.results[0].stats["auc"]
        assert actual.flags == other.flags
        assert actual.value == pytest.approx(other.value, abs=2e-4)
        for field in ["generated_distribution", "reference_distribution"]:
            if getattr(actual, field) is not None:
                np.testing.assert_allclose(
                    (
                        getattr(actual, field).samples
                        if field.endswith("_distribution")
                        else getattr(actual, field)
                    ),
                    (
                        getattr(other, field).samples
                        if field.endswith("_distribution")
                        else getattr(other, field)
                    ),
                    atol=2e-4,
                    rtol=0,
                )
        assert report.provenance["metric_execution"]["linear_c2st"]["implementation"] == "cuda_cpp"
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.set_float32_matmul_precision(before[1])
        torch.backends.cuda.matmul.allow_tf32 = before[2]
    assert torch.cuda.current_device() == before[0]


def test_source_fold_conflicts_memory_failures_and_nonconvergence_are_retained(torch, device):
    r, s, schema = frames()
    metric = LinearC2ST(device=device, solver="direct", max_iter=1, tolerance=1e-15)
    context = context_for(metric, r, s, schema)
    with pytest.raises(ValueError, match="copies"):
        context.compute_classifier(
            metric,
            np.array([[0, 0]]),
            np.array([[0, 1]]),
            np.array([[0, 1]]),
            np.array([[0, 1]]),
            [81],
        )
    with pytest.raises(ValueError, match="across classes"):
        context.compute_classifier(
            metric,
            np.array([[0, 1]]),
            np.array([[0, 1]]),
            np.array([[0, 1]]),
            np.array([[1, 0]]),
            [81],
            ideal=True,
        )
    rf, sf = source_folds(len(r), len(s), 3, np.random.default_rng(1289))
    result = context.compute_classifier(
        metric,
        np.arange(len(r))[None],
        np.arange(len(s))[None],
        rf[None],
        sf[None],
        [81],
        point=True,
    )[0]
    assert not result.converged
    assert all(v["reason"] != "decrement_squared" for v in result.folds)
    assert all(v["iterations"] <= 1 for v in result.folds)
    assert all(
        v["converged"] == (v["gradient_squared"] <= metric.tolerance**2) for v in result.folds
    )
    tiny = LinearC2ST(device=device, gpu_workspace_mb=1)
    with pytest.raises(MemoryError, match="workspace allowance"):
        context_for(tiny, r, s, schema).compute_classifier(
            tiny, np.arange(len(r))[None], np.arange(len(s))[None], rf[None], sf[None], [81]
        )


def test_cublas_current_stream_fake_shapes_and_precision_isolated(torch, device, ops):
    from torch._subclasses.fake_tensor import FakeTensorMode

    before = (
        torch.cuda.current_device(),
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    stream = torch.cuda.Stream(device=device)
    try:
        with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.float16):
            x = torch.tensor([[1.000123, 1.000321], [-0.999876, 1.000222]], device=device)
            v = torch.tensor([1.0, -1.0], device=device)
            actual = ops.mv(x, v, False)
            gram = ops.lower_gram(x)
        stream.synchronize()
        np.testing.assert_allclose(
            actual.cpu(), x.cpu().numpy().astype(float) @ v.cpu().numpy(), atol=1e-7, rtol=0
        )
        np.testing.assert_allclose(
            gram.cpu().numpy()[np.tril_indices(2)],
            (x.cpu().numpy().astype(float).T @ x.cpu().numpy())[np.tril_indices(2)],
            atol=3e-7,
            rtol=0,
        )
        assert actual.dtype == gram.dtype == torch.float32
        assert torch.backends.cuda.matmul.allow_tf32
        with FakeTensorMode() as mode:
            fx, fv = mode.from_tensor(x), mode.from_tensor(v)
            assert ops.mv(fx, fv, False).shape == (2,)
            fg = ops.lower_gram(fx)
            fl, info = ops.cholesky(fg)
            assert fl.shape == (2, 2) and info.shape == (1,)
            result, info = ops.solve(fl, fv)
            assert result.shape == (2,) and info.dtype == torch.int32
    finally:
        torch.set_float32_matmul_precision(before[1])
        torch.backends.cuda.matmul.allow_tf32 = before[2]
    assert torch.cuda.current_device() == before[0]


def test_scoring_cannot_fit_transforms_and_gpu_workspace_released(torch, device):
    from tabgen_eval.execution.torch.cuda.classifier.data import prepare

    r, s, schema = frames()
    rf, sf = source_folds(len(r), len(s), 3, np.random.default_rng(1290))
    metric = LinearC2ST(device=device, ridge=0.03, gpu_workspace_mb=32)
    args = [torch.tensor(v, device=device) for v in [rf, sf]]
    c = context_for(metric, r, s, schema)
    first = prepare(
        c.reference,
        c.candidate,
        *args,
        0,
        c.inputs.cardinalities,
        metric.native_options(point=True),
        16 * 1024**2,
        0,
    )
    heldout_r, heldout_s = r.copy(), s.copy()
    heldout_r.loc[rf == 0, "x"] = 1e12
    heldout_r.loc[rf == 0, "c"] = "heldout_only"
    heldout_s.loc[sf == 0, "y"] = -1e12
    modified = context_for(metric, heldout_r, heldout_s, schema)
    second = prepare(
        modified.reference,
        modified.candidate,
        *args,
        0,
        modified.inputs.cardinalities,
        metric.native_options(point=True),
        16 * 1024**2,
        0,
    )
    torch.testing.assert_close(first.train.dense, second.train.dense, atol=0, rtol=0)
    rd, sd = np.arange(len(r))[None], np.arange(len(s))[None]
    c.compute_classifier(metric, rd, sd, rf[None], sf[None], [81], point=True)
    # Earlier tests may leave collectable CUDA tensors. Release them before
    # measuring this call's workspace so unrelated collection cannot move it.
    gc.collect()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = c.compute_classifier(
        metric,
        np.repeat(rd, 2, 0),
        np.repeat(sd, 2, 0),
        np.repeat(rf[None], 2, 0),
        np.repeat(sf[None], 2, 0),
        [81, 81],
        point=True,
    )
    assert result[0].value == result[1].value and all(r.converged for r in result)
    torch.cuda.synchronize(device)
    assert torch.cuda.max_memory_allocated(device) - baseline <= 32 * 1024**2
    assert torch.cuda.memory_allocated(device) == baseline


def test_all_missing_training_features_retain_drop_explanations(torch, device):
    frame = pd.DataFrame({"empty": np.full(12, np.nan)})
    schema = Schema(numeric=["empty"])
    metric = LinearC2ST(device=device)
    context = context_for(metric, frame, frame.copy(), schema)
    folds = np.tile(np.arange(3), 4)[None]
    outcome = context.compute_classifier(
        metric,
        np.arange(12)[None],
        np.arange(12)[None],
        folds,
        folds,
        [21],
        point=True,
    )[0]
    assert not np.isfinite(outcome.value)
    for fold in outcome.folds:
        assert fold["features"] == 0
        assert fold["preprocessing"]["dropped_features"] == [
            {"column": "empty", "reason": "no_finite_training_values"}
        ]
