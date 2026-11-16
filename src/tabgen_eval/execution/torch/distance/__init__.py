"""Reference-fitted PRDC/DCR geometry and bounded distance execution.

``executor`` validates source draws and owns their lifetimes, ``geometry`` fits
and transforms draw-local features, ``categories`` selects category membership
storage, and ``ops`` validates registered tensor operator contracts. CUDA kernels
live under ``execution.torch.cuda.triton``; CPU kernels live under ``execution.csrc``.
Importing this namespace alone does not load either numerical backend.
"""
