"""Deterministic mixed-table fixtures shared by public API tests."""

import numpy as np
import pandas as pd


def make_frame(n: int, *, seed: int = 0, shift: float = 0.0, rare_p: float = 0.03) -> pd.DataFrame:
    """A mixed numeric/categorical frame with a deliberately rare category level."""
    rng = np.random.default_rng(seed)
    p_common = 0.5 - rare_p  # two common + two rare levels sum to 1.0
    return pd.DataFrame(
        {
            "age": rng.normal(40 + shift, 10, n).round(),
            "income": rng.gamma(2.0, 20000.0, n) + shift * 1000.0,
            "job": rng.choice(
                ["eng", "sales", "ops", "rare_role"],
                n,
                p=[p_common, p_common, rare_p, rare_p],
            ),
            "marital": rng.choice(["single", "married"], n),
        }
    )
