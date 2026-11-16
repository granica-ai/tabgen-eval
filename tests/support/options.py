"""Mapping fixtures for private numerical engine protocols."""

from tabgen_eval._options import describe_options, resolve_options


def make_options(*, uncertainty="none", n_replicates=10, **execution):
    """Resolve flat private-fixture controls through ordinary public validation.

    Numerical engine tests retain direct access to normalized internal options.
    Public API tests should pass ordinary keyword arguments to ``evaluate``.
    """
    repeated = (
        None if uncertainty == "none" else {"mode": uncertainty, "n_replicates": n_replicates}
    )
    return resolve_options(uncertainty=repeated, **execution)


def replace_options(options, **updates):
    """Copy a normalized private fixture and validate the updated controls."""
    return make_options(**(dict(options) | updates))


def public_kwargs(options):
    """Expose a shared internal fixture through the evaluator's public keywords."""
    return describe_options(options)
