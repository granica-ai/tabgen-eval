"""CUDA-specific fidelity orchestration bridges.

Metric definitions and shared tensor preparation remain in ``execution.torch.fidelity``.
This namespace contains the optional native launcher bridge, which borrows
compiled Triton functions while Python retains their module owners. Importing
this namespace alone does not import Torch or load the native extension.
"""
