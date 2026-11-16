"""Evaluation-local preparation, tensor ownership and execution routing.

Evaluation coordinators use ``ExecutionSession`` and numerical context contracts
to share source storage and submit assigned observations. Source-role planning,
uncertainty collection, and report assembly remain with their respective callers.
Host preparation and optional Arrow packing are separate from metric-specific
fitted transforms. Torch/device operations remain lazy behind tensor/context
helpers; importing this package alone loads no numerical backend or CUDA state."""
