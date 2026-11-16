"""CUDA execution of the shared classifier protocol.

``executor`` owns draw/fold lifetimes and ``data`` fits training-only transforms.
``solver`` dispatches to thin ``newton``/``saga`` packet adapters. Compiled CUDA
code owns Newton gradients, damping and Armijo iteration, plus SAGA preparation,
preconditioning and update kernels. Python retains SAGA scheduling, rollback and
diagnostics. Importing this namespace alone does not import Torch or initialize
CUDA.
"""
