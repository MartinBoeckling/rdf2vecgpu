"""Monkey-patches for cuGraph 25.6 + dask 2025.5 compatibility.

Known upstream bugs worked around here:
1. ``dask_cudf.from_cudf()`` broken with dask-expr backend
2. ``convert_to_cudf`` in cuGraph's dask uniform_random_walks crashes
   when ``number_map=None`` (i.e. ``renumber=False``)

These patches are applied once at import time via ``apply_patches()``.
"""

import math
import sys
from loguru import logger

_PATCHED = False


def apply_patches():
    """Apply compatibility patches (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        _patch_dask_cudf_from_cudf()
    except Exception as e:
        logger.debug(f"Skipping dask_cudf.from_cudf patch: {e}")

    try:
        _patch_convert_to_cudf()
    except Exception as e:
        logger.debug(f"Skipping convert_to_cudf patch: {e}")


def _patch_dask_cudf_from_cudf():
    """Replace ``dask_cudf.from_cudf`` with a ``from_delayed``-based version.

    The dask-expr backend (default in dask >=2025.5) breaks the original
    ``from_cudf`` for cudf DataFrames/Series.
    """
    import dask
    import cudf
    import dask_cudf

    def _fixed_from_cudf(data, npartitions=1):
        if isinstance(data, (dask_cudf.DataFrame, dask_cudf.Series)):
            return data
        if isinstance(data, (cudf.DataFrame, cudf.Series)):
            n = len(data)
            chunk = max(1, math.ceil(n / npartitions))
            parts = [data.iloc[i : i + chunk] for i in range(0, n, chunk)]
            return dask_cudf.from_delayed([dask.delayed(p) for p in parts])
        raise TypeError(f"Expected cudf object, got {type(data)}")

    dask_cudf.from_cudf = _fixed_from_cudf
    logger.debug("Patched dask_cudf.from_cudf for dask-expr compatibility")


def _patch_convert_to_cudf():
    """Fix ``convert_to_cudf`` in cuGraph's dask random walk modules.

    When the graph is created with ``renumber=False``, ``number_map`` is
    ``None`` and the original code crashes with an AttributeError.
    Applies to both uniform_random_walks and biased_random_walks.
    """
    import cudf
    import cugraph.dask.sampling.uniform_random_walks as urw_mod
    import cugraph.dask.sampling.biased_random_walks as brw_mod

    def _fixed_convert_to_cudf(cp_paths, number_map=None, is_vertex_paths=False):
        if is_vertex_paths and len(cp_paths) > 0 and number_map is not None:
            if number_map.implementation.numbered:
                df_ = cudf.DataFrame()
                df_["vertex_paths"] = cp_paths
                df_ = number_map.unrenumber(
                    df_, "vertex_paths", preserve_order=True
                ).compute()
                return cudf.Series(df_["vertex_paths"]).fillna(-1)
        return cudf.Series(cp_paths)

    urw_mod.convert_to_cudf = _fixed_convert_to_cudf
    urw_mod.dask_cudf = sys.modules.get("dask_cudf", urw_mod.dask_cudf)

    brw_mod.convert_to_cudf = _fixed_convert_to_cudf
    brw_mod.dask_cudf = sys.modules.get("dask_cudf", brw_mod.dask_cudf)

    logger.debug("Patched cugraph convert_to_cudf for renumber=False compatibility")
