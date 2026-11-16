"""Device kernels and their allocation/launch wrappers for tensor CUDA execution.

Individual modules import Torch/Triton only when selected by execution routing;
importing this namespace alone neither compiles kernels nor initializes CUDA.
Fidelity kernels implement selected-precision products and exact identity counts;
distance kernels provide metric geometry, while classifier kernels support the
native training/preparation paths. Host-side validation belongs to registered
operator adapters; launch wrappers own output/scratch allocation and use the
input device's current stream. Kernel comments record layouts and masking rules,
not public report semantics.
"""
