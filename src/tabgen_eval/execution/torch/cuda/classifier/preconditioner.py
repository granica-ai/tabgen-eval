"""Wrap native fixed Nyström curvature and its factored inverse.

Production ``build`` delegates diagonal/low-rank construction, power estimation,
row calibration, and factorization to C++/CUDA. Both low-rank and residual-
diagonal terms use the same curvature rows when large-fit sampling is enabled.
Python keeps factor ownership and serializable diagnostics, not the numerical
preparation loop. The power estimate and doubled margins are empirical rather
than certified minibatch spectral bounds.

``CurvatureFactor.solve``/``solve_stable`` expose Python numerical references
used by oracles. Production training applies the factors inside native SAGA;
its FP64 path uses the cached basis and its FP32 path may select Householder
coordinates to avoid cancellation at small inverse ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from tabgen_eval.execution.torch.cuda.classifier.device import operators
from tabgen_eval.execution.torch.cuda.classifier.packets import (
    PreparationDiagnostics,
    PreparationInputs,
)
from tabgen_eval.execution.torch.cuda.extension import library

if TYPE_CHECKING:
    from tabgen_eval.execution.torch.cuda.classifier.data import Design


def _product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # FP64 retains the verified Torch trajectory. FP32 factor construction
    # obeys declared precision even when callers enable ambient TF32.
    """Multiply factor matrices without ambient TF32 changing FP32 arithmetic.

    Matching CUDA inputs follow ordinary matrix multiplication shapes. FP32 uses
    the native pedantic GEMM; FP64 preserves the established Torch matmul path.
    Return a fresh tensor without changing either operand. Host-library loading
    is deferred until the FP32 branch is used.
    """
    return library().gemm(a, b) if a.dtype == torch.float32 else a @ b


@dataclass
class CurvatureFactor:
    """Own a diagonal-plus-low-rank inverse representation on one CUDA device.

    ``inverse_root`` is selected-precision ``[D]``. Column-major ``reflectors``
    ``[D, R]`` and ``tau`` ``[R]`` encode Q; ``rotations`` ``[R, R]`` and
    ``ratios`` ``[R]`` represent its low-rank inverse correction. Construction
    retains supplied tensor references. ``_basis`` lazily owns the column-major
    product Q @ rotations, normally supplied directly by native preparation.
    The representation includes the intercept coordinate and is fixed for a fit.
    """

    inverse_root: torch.Tensor
    reflectors: torch.Tensor
    tau: torch.Tensor
    rotations: torch.Tensor
    ratios: torch.Tensor
    _basis: torch.Tensor | None = field(default=None, init=False, repr=False)

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """Apply the represented inverse to a matching CUDA vector ``rhs[D]``.

        Return a fresh vector while treating factors/RHS as read-only. FP64 uses
        the cached basis Woodbury form, with no Householder application. FP32 uses
        the stable Householder reference below. This helper supports numerical
        oracles; production minibatch updates apply the same factors in native code.
        The first FP64 call may populate the instance's basis cache.
        """
        if rhs.dtype == torch.float64:
            basis = self.basis()
            scaled = self.inverse_root * rhs
            return self.inverse_root * (scaled + basis @ ((self.ratios - 1) * (basis.T @ scaled)))
        return self.solve_stable(rhs)

    def solve_stable(self, rhs: torch.Tensor) -> torch.Tensor:
        """Apply the inverse through explicit orthogonal Householder coordinates.

        ``rhs`` is a selected-precision CUDA ``[D]`` vector matching the factors.
        Two ``torch.ormqr`` calls avoid subtracting nearly equal Woodbury terms;
        new coordinate/output buffers are allocated and factors/RHS stay read-only.
        This Python reference also backs ``solve`` for FP32. Production SAGA uses
        its native equivalent when FP32 inverse ratios require the safeguard; FP64
        production never calls this method.
        """
        coordinates = torch.ormqr(
            self.reflectors, self.tau, (self.inverse_root * rhs)[:, None], transpose=True
        )
        rank = len(self.ratios)
        projection = self.rotations.T @ coordinates[:rank, 0]
        coordinates[:rank, 0] = self.rotations @ (self.ratios * projection)
        return (
            self.inverse_root
            * torch.ormqr(self.reflectors, self.tau, coordinates, transpose=False)[:, 0]
        )

    def basis(self) -> torch.Tensor:
        """Return the retained column-major CUDA ``[D, R]`` low-rank basis.

        On a cache miss, form Q from the packed QR factors and multiply by the
        small rotations, then store the resulting tensor on this instance. Native
        preparation ordinarily fills this cache, so training incurs no reconstruction.
        The returned object is shared factor storage and callers must not mutate it.
        """
        if self._basis is None:
            self._basis = (
                _product(torch.orgqr(self.reflectors, self.tau), self.rotations).T.contiguous().T
            )
        return self._basis


def build(
    design: Design,
    ridge: float,
    rank: int,
    generator: torch.Generator,
    scratch_limit: int,
    *,
    sampled: bool = True,
    preparation_rows: int = 4096,
    batch: int = 4096,
) -> tuple[CurvatureFactor, float, dict[str, object]]:
    """Construct fixed curvature and calibrate the SAGA rate in native C++/CUDA.

    Parameters
    ----------
    design : Design
        Training-only compact CUDA design in FP32/FP64. Missing inverse category
        segments are built and cached on the design before the native call.
    ridge : float
        Feature penalty in the full-training average-loss objective; intercept
        remains unpenalized and sampling does not change its normalization.
    rank : int
        Requested Nyström rank, capped at the complete design width.
    generator : torch.Generator
        Private generator on the design device for Gaussian sketch/power draws.
    scratch_limit : int
        Byte allowance for native curvature factors and temporary workspaces.
    sampled : bool, default=True
        Permit coherent row sampling only at the native 500,000-training-row gate.
    preparation_rows : int, default=4096
        Maximum sampled curvature rows after that gate; otherwise use all rows.
    batch : int, default=4096
        Actual minibatch size used by the expected-smoothness rate calibration.

    Returns
    -------
    tuple of CurvatureFactor, float, dict
        Owned factors with a cached basis, the calibrated positive learning rate,
        and host diagnostics for factorization, row sampling, and curvature bounds.

    Raises
    ------
    ImportError
        The selected host library lacks native preparation operators.
    torch.OutOfMemoryError
        Native retained/temporary storage exceeds the supplied allowance.
    torch.linalg.LinAlgError
        Curvature is invalid; recognized factorization failures may instead use a
        positive diagonal fallback, recorded explicitly in returned diagnostics.

    Notes
    -----
    The native sampler derives a separate stream from the generator's seed;
    sketch/power draws may advance that private generator. Input design buffers
    are read-only except for caching inverse row lists. Only curvature is sampled:
    training, vocabulary, transforms, stopping estimates, and scoring remain full.
    """
    if design.inverse_offsets is None:
        design.invert()
    ops, ptx = operators(design.numbers)
    if not hasattr(ops, "saga_prepare"):
        raise ImportError(
            "The selected CUDA library lacks native SketchySAGA preparation; rebuild it."
        )
    result = ops.saga_prepare(
        list(
            PreparationInputs(
                numbers=design.numbers,
                categories=design.identities,
                category_centers=design.centers,
                category_scales=design.scales,
                category_rows=design.inverse_rows,
                category_offsets=design.inverse_offsets,
            )
        ),
        ptx,
        float(ridge),
        min(int(rank), design.d),
        generator,
        int(scratch_limit),
        bool(sampled),
        int(preparation_rows),
        int(batch),
    )
    inverse_root, reflectors, tau, rotations, ratios, basis, diagnostics = result
    factor = CurvatureFactor(inverse_root, reflectors, tau, rotations, ratios)
    factor._basis = basis
    values = PreparationDiagnostics._make(diagnostics.tolist())
    metadata = dict(
        preconditioner="diagonal_fallback" if values.fallback else "nystrom_residual_diagonal",
        preconditioner_rank=0 if values.fallback else len(ratios),
        factorization_fallbacks=int(values.fallback),
        estimated_eigenvalue=values.data_curvature,
        learning_rate=values.learning_rate,
        eigenvalue_estimator="30 power iterations; empirical factor-of-two margin",
        residual_diagonal_minimum=values.residual_diagonal_min,
        minimum_inverse_ratio=values.inverse_ratio_min,
        preparation_training_rows=int(values.full_rows),
        preparation_rows=int(values.curvature_rows),
        sampled_preparation=bool(values.sampled),
        preparation_sample_hash=int(values.sample_hash),
        preparation_sampling_threshold=500000,
        learning_rate_calibration="Gazagnadou practical expected smoothness; empirical power/sample margins",
        curvature_data_raw=float(values.data_curvature_raw),
        curvature_data_used=float(values.data_curvature_used),
        curvature_row_maximum_raw=float(values.row_maximum_raw),
        curvature_row_maximum_used=float(values.row_maximum_used),
        curvature_ridge_bound=float(values.ridge_bound),
        curvature_practical=float(values.practical_curvature),
        curvature_calibration_rows=int(values.calibration_rows),
        curvature_calibration_population=int(values.calibration_population),
        curvature_calibration_batch=int(values.calibration_batch),
    )
    return factor, values.learning_rate, metadata
