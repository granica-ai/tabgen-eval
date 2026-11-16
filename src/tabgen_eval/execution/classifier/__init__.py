"""Backend-independent Linear C2ST source-fold and seed contracts.

``protocol`` keeps each original source row in one held-out fold and preserves
exact uint64 random seeds. Python CPU fitting lives in ``execution.python``;
compiled CPU fitting lives in native C++. CUDA preparation and solver adapters
live under ``execution.torch.cuda.classifier``. Importing this namespace loads
no classifier implementation."""
