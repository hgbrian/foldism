"""Backend prediction functions for AlphaFold 2, ESMFold 2, OpenDDE, and Mol-1.

Each backend downloads models to a persistent Modal Volume (`foldism-models`)
on first build under per-backend subdirs (e.g. `/models/mol1/`).
Each backend selects its own GPU: ESMFold 2 and OpenDDE use A100-40GB; Mol-1 and AlphaFold 2 use A100-80GB.
"""

from .common import (
    CACHE_VOLUME,
    FOLDING_APPS,
    app,
    check_cache,
    convert_for_app,
    get_cache_key,
    get_cache_subdir,
)
from .alphafold2 import alphafold_predict
from .colabsearch import colabsearch_fetch
from .esmfold2 import esmfold2_predict
from .opendde import opendde_predict
from .mol1 import mol1_predict

__all__ = [
    "CACHE_VOLUME",
    "FOLDING_APPS",
    "alphafold_predict",
    "app",
    "check_cache",
    "colabsearch_fetch",
    "convert_for_app",
    "esmfold2_predict",
    "get_cache_key",
    "get_cache_subdir",
    "opendde_predict",
    "mol1_predict",
]
