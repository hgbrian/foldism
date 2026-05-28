"""Backend prediction functions for Boltz-2, Chai-1, Protenix, AlphaFold 2, OpenFold 3, and ESMFold 2.

Each backend downloads models to a persistent Modal Volume (`foldism-models`)
on first build under per-backend subdirs (e.g. `/models/boltz2/`, `/models/chai1/`).
GPU is configurable via GPU environment variable (default: L40S, Chai-1/AF2 use A100-80GB).
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
from .boltz import boltz2_predict
from .chai1 import chai1_predict
from .colabsearch import colabsearch_fetch
from .esmfold2 import esmfold2_predict
from .openfold3 import openfold3_predict
from .protenix import protenix_predict

__all__ = [
    "CACHE_VOLUME",
    "FOLDING_APPS",
    "alphafold_predict",
    "app",
    "boltz2_predict",
    "chai1_predict",
    "check_cache",
    "colabsearch_fetch",
    "convert_for_app",
    "esmfold2_predict",
    "get_cache_key",
    "get_cache_subdir",
    "openfold3_predict",
    "protenix_predict",
]
