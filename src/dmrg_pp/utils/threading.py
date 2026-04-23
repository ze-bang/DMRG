"""Thread-pool management for the linear-algebra backends.

DMRG sweeps issue *many* small BLAS calls per second.  When the underlying
LAPACK/BLAS implementation (OpenBLAS, MKL, Accelerate, …) is configured with
its default thread pool — typically the number of physical cores — the
overhead of parking and waking threads dominates the actual flops on the
small per-bond tensor blocks (D ≲ 200, d = 2…4).  The end-to-end speed-up
from forcing single-threaded BLAS for these workloads can be 10×–100× on
hyper-threaded machines and inside container/WSL environments.

This module provides:

* :func:`limit_blas_threads`   — a process-wide knob that pins the BLAS
  thread pool via :mod:`threadpoolctl` if available, otherwise falls back to
  the well-known environment variables (in case they are inspected
  per-import by some backends).
* :func:`single_threaded_blas` — a context manager wrapping a code block.

Both are no-ops if neither :mod:`threadpoolctl` is installed nor the
environment variables are honoured.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

__all__ = ["limit_blas_threads", "single_threaded_blas"]


_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def limit_blas_threads(n: int = 1) -> None:
    """Cap the BLAS thread pool to ``n`` worker threads, process-wide."""
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=n)
    except ImportError:
        pass
    for var in _BLAS_ENV_VARS:
        os.environ.setdefault(var, str(n))


@contextlib.contextmanager
def single_threaded_blas() -> Iterator[None]:
    """Temporarily restrict BLAS to a single thread for the wrapped block."""
    try:
        import threadpoolctl

        with threadpoolctl.threadpool_limits(limits=1):
            yield
    except ImportError:
        yield
