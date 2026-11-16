"""Compiled ATen CPU loading, draw dispatch, and classifier result decoding.

``extension`` registers the native operators lazily; ``executor`` adapts prepared
tensors and estimator-owned draws to those operators; ``diagnostics`` translates
the numeric wire format into report fields. Numerical implementations and
registration translation units are owned by ``execution/csrc/cpu``.
"""
