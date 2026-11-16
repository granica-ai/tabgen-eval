"""Precision-preserving Shape, Trend and Conditional tensor execution.

Executors consume prepared source pools and explicit source-row draws; they do
not choose bootstrap samples or construct confidence intervals. Shared primitives
implement stable finite-cell correlations, reference-fitted bins and exact
category identities. Registered primitives separate the algorithms from CUDA
launch code; native CPU report execution enters through the CPU adapter. Importing this
namespace alone does not import Torch or initialize a CUDA device.
"""
