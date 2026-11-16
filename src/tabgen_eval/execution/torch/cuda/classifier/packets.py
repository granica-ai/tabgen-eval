"""Named CUDA operator packets and immutable wire-format diagnostics.

Field order mirrors ``csrc/cuda/classifier/packets.hpp``. Conversion to lists
is confined to Torch operator calls, and diagnostic tuples decode small CPU
arrays returned by native execution. These containers hold references only;
immutability of a tuple does not make its tensor storage immutable. Native fit
and preparation operators treat the supplied buffers as read-only.
"""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class FitInputs(NamedTuple):
    """Borrow the fifteen tensors needed by one native SAGA chunk.

    ``numbers`` is contiguous FP32/FP64 CUDA ``[N, PN]``; ``categories`` is
    contiguous int64 ``[N, PC]`` with compact feature IDs or unknown ``-1``.
    Category center/scale vectors have length K, so ``D = PN + K + 1``.
    Labels are selected-precision ``[N]`` and schedule is int32 ``[E, S, B]``.
    Initial weights/average are ``[D]`` and the scalar loss-gradient table is
    ``[N]``. The operator returns new state buffers, enabling reference rollback.

    Inverse factors share dtype/device. QR reflectors, small rotations, and
    cached basis use column-major matrices ``[D, R]``, ``[R, R]``, and ``[D, R]``;
    inverse_root has length D and tau/ratios length R. In Householder mode ratios
    are inverse eigenvalue ratios; basis mode supplies ratios minus one. The
    separate scalar inverse-mode argument determines which fields are consumed.
    """

    numbers: Tensor
    categories: Tensor
    category_centers: Tensor
    category_scales: Tensor
    labels: Tensor
    schedule: Tensor
    inverse_root: Tensor
    reflectors: Tensor
    tau: Tensor
    rotations: Tensor
    ratios: Tensor
    basis: Tensor
    initial_weights: Tensor
    initial_table: Tensor
    initial_average: Tensor


class PreparationInputs(NamedTuple):
    """Borrow the six compact-design tensors for native curvature preparation.

    Numbers/identities are row-major CUDA ``[N, PN]`` FP32/FP64 and ``[N, PC]``
    int64, respectively. Selected-precision category centers/scales have length
    K and retain the training-fitted full vocabulary. Int64 category_rows lists
    stable observation segments; category_offsets has length K+1. Sampling may
    replace native working copies of rows and segments without changing these
    inputs or the full training population used to normalize ridge and the rate.
    """

    numbers: Tensor
    categories: Tensor
    category_centers: Tensor
    category_scales: Tensor
    category_rows: Tensor
    category_offsets: Tensor


class PreparationDiagnostics(NamedTuple):
    """Decode the eighteen FP64 host scalars returned by native preparation.

    Flags, counts, and the exactly representable sample hash travel as doubles
    in the native ABI and are converted to their semantic types by ``build``.
    The first curvature field is the safety-adjusted estimate; raw/used data and
    row maxima are also reported separately. Ridge bound and practical curvature
    explain rate calibration, while full/curvature/calibration row counts expose
    sampling. This tuple owns no factors and provides no convergence certificate.
    """

    fallback: float
    data_curvature: float
    learning_rate: float
    residual_diagonal_min: float
    inverse_ratio_min: float
    full_rows: float
    curvature_rows: float
    sampled: float
    sample_hash: float
    data_curvature_raw: float
    data_curvature_used: float
    row_maximum_raw: float
    row_maximum_used: float
    ridge_bound: float
    practical_curvature: float
    calibration_rows: float
    calibration_population: float
    calibration_batch: float


class EpochHistory(NamedTuple):
    """Decode one six-double native SAGA epoch summary copied to host memory.

    Maximum coefficient/change exclude the intercept; intercept_abs_change is
    reported separately. ``finite`` is a numeric boolean. Gradient infinity norms
    refer to existing unpreconditioned online SAGA directions, including feature
    ridge: current immediately before the last update and initial before the
    first update. They are not exact full-data gradients or preconditioned norms.
    """

    maximum_abs_coefficient: float
    maximum_abs_change: float
    finite: float
    intercept_abs_change: float
    online_gradient_infinity: float
    initial_gradient_infinity: float
