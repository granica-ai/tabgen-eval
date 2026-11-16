"""Small independent dense classifier oracles shared by CPU tests."""

import numpy as np
from scipy.special import expit


def design(torch, n=123):
    rng = np.random.default_rng(93)
    x = np.column_stack([rng.normal(size=(n, 4)), np.ones(n)])
    y = rng.binomial(1, expit(x @ np.array([0.8, -0.3, 0.2, 0.7, -0.1])))
    return torch.from_numpy(x), torch.from_numpy(y.astype(float))


def independent_loss(x, y, w, ridge):
    z = x @ w
    return np.logaddexp(0, np.where(y == 1, -z, z)).mean() + 0.5 * ridge * np.sum(w[:-1] ** 2)


def fit(ops, x, y, solver, *, max_iter=100, seed=31, threads=1):
    return ops.classifier_fit(
        x, y, solver, 0.03, 1e-8, max_iter, 500, 5e10, 500, 4096, 64, seed, threads
    )
