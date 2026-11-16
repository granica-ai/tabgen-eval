"""CUDA adapters, native-library loading, and Triton launch wrappers.

This namespace itself has no Torch import or CUDA initialization side effects.
Classifier/fidelity adapters enter native host operators through ``extension``;
``kernels`` owns the Triton operations and ``host`` distributes their C launchers.
Native source ownership and build inventories live in ``execution.csrc``.
"""
