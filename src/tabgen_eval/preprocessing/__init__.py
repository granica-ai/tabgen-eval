"""Shared source storage and training-fitted CPU/CUDA preprocessing.

Python owns policy and dispatch; compiled backends own numerical transforms.
Callers declare representations; metrics own fitted transforms and missingness rules.
"""

from typing import TYPE_CHECKING

from .contracts import POLICY_VERSION, CategoryState, NumericState, PreparedFold

if TYPE_CHECKING:
    # Type checkers see the real signatures; runtime wrappers preserve the
    # import boundary without duplicating those signatures and defaults.
    from .folds import fit_fold as fit_fold
    from .sources import prepare_sources as prepare_sources
else:

    def prepare_sources(*args, **kwargs):
        """Prepare reusable sources without importing a numerical backend eagerly."""
        from .sources import prepare_sources as prepare

        return prepare(*args, **kwargs)

    def fit_fold(*args, **kwargs):
        """Fit training-only state and transform one assigned classifier fold."""
        from .folds import fit_fold as fit

        return fit(*args, **kwargs)


__all__ = [
    "POLICY_VERSION",
    "CategoryState",
    "NumericState",
    "PreparedFold",
    "fit_fold",
    "prepare_sources",
]
