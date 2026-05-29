"""Backend adapters.

Each backend is a thin wrapper that adapts an external quantum / photonic
library (PennyLane, MerLin, DeepQuantum, ...) to Qaithon's :class:`Backend`
contract. Importing this package auto-registers all bundled backends.

Importing this module is intentionally cheap: heavy optional dependencies
are only imported inside their respective backend modules, which are loaded
lazily by the registry on first ``get_backend(...)`` call.
"""

from __future__ import annotations

from qaithon.backends.base import (
    Backend,
    BackendFactory,
    BackendProfile,
    BackendRegistry,
    get_backend,
    list_backends,
    register_backend,
)

# Side-effect import: registers the "mock" backend in the default registry.
# Imported eagerly because it has no heavy dependencies and is needed by tests.
from qaithon.backends import mock as _mock  # noqa: F401
from qaithon.backends import pennylane_backend as _pennylane  # noqa: F401
from qaithon.backends import quandela_sim as _quandela_sim  # noqa: F401
from qaithon.backends import deepquantum_backend as _deepquantum  # noqa: F401
from qaithon.backends import ibm_aer as _ibm_aer  # noqa: F401
from qaithon.backends import ibm_heron as _ibm_heron  # noqa: F401
from qaithon.backends import quandela_belenos as _quandela_belenos  # noqa: F401
from qaithon.backends import aws_braket_sv1 as _aws_braket_sv1  # noqa: F401
from qaithon.backends import perceval_photonic as _perceval_photonic  # noqa: F401
from qaithon.backends import quandela_merlin as _quandela_merlin  # noqa: F401
from qaithon.backends import aws_braket_qpus as _aws_braket_qpus  # noqa: F401

# Eagerly discover third-party plugin backends declared via entry_points.
# Failures here are isolated and logged — never crash on import.
try:
    from qaithon import plugins as _plugins

    _plugins.discover()
except Exception:  # noqa: BLE001
    pass

__all__ = [
    "Backend",
    "BackendFactory",
    "BackendProfile",
    "BackendRegistry",
    "get_backend",
    "list_backends",
    "register_backend",
]
