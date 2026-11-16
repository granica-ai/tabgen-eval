"""Internal preparation, dispatch, and numerical execution for metric estimators.

``python`` owns NumPy/SciPy calculations and statistical helpers.
``runtime`` owns eligible source pools, shared outcomes, and backend selection;
``xgboost`` owns shared CPU/CUDA model fitting and prediction;
``classifier`` defines shared source-fold and seed contracts. ``torch`` owns
tensor storage and CPU/CUDA execution adapters. Their C++ and CUDA implementation
sources live under ``csrc``. Public metric and estimator APIs live outside this
package. Importing this namespace alone loads no tensor or device backend.
"""
