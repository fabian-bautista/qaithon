"""AWS Braket backends for real QPUs (QuEra Aquila, IonQ Forte).

Both backends share the same scaffolding as
:class:`AWSBraketSV1Backend`: ``mode="profile"`` costs nothing and runs
``F.linear``, ``mode="calibrate"`` dispatches one real calibration
circuit per forward call (consumes shots → real AWS spend).

These are **opt-in for real benchmarks only**. The default mode is
profile so accidentally compiling a model against ``aws.braket.quera``
does not bill anyone.
"""

from __future__ import annotations

import importlib.util
import time
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.backends._realhw_common import BackendMode, RealHardwareBackendBase
from qaithon.backends.base import BackendProfile, register_backend
from qaithon.config import get_aws_credentials

if TYPE_CHECKING:
    pass

__all__ = ["AWSBraketIonQBackend", "AWSBraketQuEraBackend"]

logger = get_logger(__name__)


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _make_aws_session():  # type: ignore[no-untyped-def]
    from braket.aws import AwsSession
    import boto3

    access_id, secret, region = get_aws_credentials()
    boto_session = boto3.Session(
        aws_access_key_id=access_id,
        aws_secret_access_key=secret,
        region_name=region,
    )
    return AwsSession(boto_session=boto_session)


# ---------------------------------------------------------------------------
# QuEra Aquila — neutral-atom QPU
# ---------------------------------------------------------------------------
_QUERA_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"


class AWSBraketQuEraBackend(RealHardwareBackendBase):
    """QuEra Aquila (neutral atom) via AWS Braket.

    Programming model is analog — Hamiltonian schedule rather than gates.
    For our purposes we use a tiny placeholder schedule that exercises
    the cloud path and reports actual latency. Real production usage
    requires a proper Hamiltonian definition.

    Args:
        mode: Operation mode (``"profile"`` | ``"calibrate"`` | ``"execute"``).
        shots: Shots per calibration call. Default 100.
    """

    profile: BackendProfile = BackendProfile(
        name="aws.braket.quera",
        kind="quantum",
        energy_pj_per_mac=0.02,  # Neutral-atom optical control is power-efficient.
        latency_us_per_op=300_000.0,  # ~300 ms typical per shot batch.
        queue_us=10_000_000.0,  # ~10 s typical queue.
        supports_autograd=False,
        supports_batching=False,
        max_dim=256,  # Aquila qubit count.
        notes=(
            "QuEra Aquila — neutral-atom analog QPU. ~256 atoms in 2D lattice. "
            "Programming model is analog (Hamiltonian schedule), not gate-based. "
            "Pay-per-shot via AWS Braket."
        ),
    )

    def __init__(self, mode: BackendMode = "profile", shots: int = 100) -> None:
        super().__init__(mode=mode)
        if shots < 1:
            raise ValueError(f"shots must be positive, got {shots}.")
        self._shots = shots
        self._device = None  # lazy
        from qaithon.metrics import NeutralAtomMetrics
        self._last_neutral_atom_metrics: NeutralAtomMetrics | None = None

    def is_available(self) -> bool:
        if not _has("braket"):
            return False
        access_id, secret, _ = get_aws_credentials()
        return bool(access_id and secret)

    @property
    def last_neutral_atom_metrics(self):  # type: ignore[no-untyped-def]
        """Neutral-atom metrics from the most recent ``calibrate`` shot.

        Returns ``None`` if no shot has been fired (e.g. in ``profile``
        mode), otherwise a :class:`qaithon.metrics.NeutralAtomMetrics`
        instance with atom count, Rabi-style noise proxy and Rydberg
        blockade radius.
        """
        return self._last_neutral_atom_metrics

    def health_check(self):  # type: ignore[no-untyped-def]
        from qaithon.backends.base import HealthStatus
        import time as _t

        if not self.is_available():
            return HealthStatus(backend="aws.braket.quera", online=False,
                                message="braket SDK or AWS credentials missing")
        t0 = _t.perf_counter()
        try:
            device = self._get_device()
            status_str = getattr(device, "status", "UNKNOWN")
            online = str(status_str).upper() == "ONLINE"
            return HealthStatus(
                backend="aws.braket.quera", online=online,
                message=f"QuEra Aquila status={status_str}",
                latency_ms=(_t.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(backend="aws.braket.quera", online=False,
                                message=f"{type(exc).__name__}: {exc}",
                                latency_ms=(_t.perf_counter() - t0) * 1000.0)

    def _get_device(self):  # type: ignore[no-untyped-def]
        if self._device is not None:
            return self._device
        from braket.aws import AwsDevice

        self._device = AwsDevice(_QUERA_ARN, aws_session=_make_aws_session())
        return self._device

    def _calibrate_once(self) -> float:
        """Submit a minimal Aquila program and read the dispatch latency."""
        from braket.ahs import AnalogHamiltonianSimulation, AtomArrangement
        from braket.ahs import DrivingField, ShiftingField
        from braket.timings.time_series import TimeSeries

        from qaithon.metrics import NeutralAtomMetrics

        # 3-atom triangle, simplest possible AHS program.
        register = AtomArrangement()
        register.add((0.0, 0.0))
        register.add((4e-6, 0.0))
        register.add((2e-6, 4e-6))
        n_atoms = 3
        # Rydberg blockade radius ≈ 8 µm for Aquila at typical detunings.
        rydberg_radius_um = 8.0

        time_series = TimeSeries()
        time_series.put(0.0, 0.0).put(1e-6, 0.0)
        driving_field = DrivingField(
            amplitude=time_series, detuning=time_series, phase=time_series
        )
        ahs = AnalogHamiltonianSimulation(
            register=register, hamiltonian=driving_field
        )

        device = self._get_device()
        t0 = time.perf_counter()
        task = device.run(ahs, shots=self._shots)
        result = task.result()
        self._last_circuit_latency_us = (time.perf_counter() - t0) * 1e6
        noise = max(0.005, self._noise_strength_proxy())

        # Rearrangement success rate from the AHS metadata when present.
        rearrangement_rate: float | None = None
        try:
            measurements = getattr(result, "measurements", None) or []
            successes = sum(
                1 for m in measurements
                if getattr(m, "status", "Success") == "Success"
            )
            rearrangement_rate = successes / max(1, len(measurements))
        except Exception:  # noqa: BLE001
            rearrangement_rate = None

        self._last_neutral_atom_metrics = NeutralAtomMetrics(
            backend="aws.braket.quera",
            latency_us=self._last_circuit_latency_us,
            estimated_energy_pj=self.profile.energy_pj_per_mac * self._shots,
            fidelity=max(0.0, 1.0 - noise),
            n_atoms_used=n_atoms,
            rearrangement_success_rate=rearrangement_rate,
            rydberg_blockade_radius_um=rydberg_radius_um,
        )
        return noise

    def _noise_strength_proxy(self) -> float:
        return 0.005


# ---------------------------------------------------------------------------
# IonQ Forte — trapped-ion QPU
# ---------------------------------------------------------------------------
_IONQ_FORTE_ARN = (
    "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1"
)


class AWSBraketIonQBackend(RealHardwareBackendBase):
    """IonQ Forte Enterprise 1 (trapped ion) via AWS Braket.

    Trapped-ion QPUs have all-to-all connectivity and high fidelity but
    slow gate times. Per-circuit latency dominates over queue time when
    queue is short.

    Args:
        mode: ``"profile"`` | ``"calibrate"`` | ``"execute"``.
        shots: Default 32.
    """

    profile: BackendProfile = BackendProfile(
        name="aws.braket.ionq",
        kind="quantum",
        energy_pj_per_mac=0.05,  # Lasers + vacuum systems are power-heavy.
        latency_us_per_op=60_000_000.0,  # ~60 s per circuit (queue + slow gates).
        queue_us=5_000_000.0,
        supports_autograd=False,
        supports_batching=False,
        max_dim=36,  # Forte Enterprise 1 qubit count.
        notes=(
            "IonQ Forte Enterprise 1 — trapped ion QPU. 36 qubits, all-to-all "
            "connectivity. Slow but high-fidelity. Pay-per-shot."
        ),
    )

    def __init__(self, mode: BackendMode = "profile", shots: int = 32) -> None:
        super().__init__(mode=mode)
        if shots < 1:
            raise ValueError(f"shots must be positive, got {shots}.")
        self._shots = shots
        self._device = None  # lazy

    def is_available(self) -> bool:
        if not _has("braket"):
            return False
        access_id, secret, _ = get_aws_credentials()
        return bool(access_id and secret)

    def health_check(self):  # type: ignore[no-untyped-def]
        from qaithon.backends.base import HealthStatus
        import time as _t

        if not self.is_available():
            return HealthStatus(backend="aws.braket.ionq", online=False,
                                message="braket SDK or AWS credentials missing")
        t0 = _t.perf_counter()
        try:
            device = self._get_device()
            status_str = getattr(device, "status", "UNKNOWN")
            online = str(status_str).upper() == "ONLINE"
            return HealthStatus(
                backend="aws.braket.ionq", online=online,
                message=f"IonQ Forte status={status_str}",
                latency_ms=(_t.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(backend="aws.braket.ionq", online=False,
                                message=f"{type(exc).__name__}: {exc}",
                                latency_ms=(_t.perf_counter() - t0) * 1000.0)

    def _get_device(self):  # type: ignore[no-untyped-def]
        if self._device is not None:
            return self._device
        from braket.aws import AwsDevice

        self._device = AwsDevice(_IONQ_FORTE_ARN, aws_session=_make_aws_session())
        return self._device

    def _calibrate_once(self) -> float:
        from braket.circuits import Circuit

        circuit = Circuit().h(0).cnot(0, 1).cnot(1, 2)
        device = self._get_device()

        t0 = time.perf_counter()
        task = device.run(circuit, shots=self._shots)
        result = task.result()
        self._last_circuit_latency_us = (time.perf_counter() - t0) * 1e6

        counts = result.measurement_counts
        total = sum(counts.values())
        ideal = {"000", "111"}
        unexpected = sum(c for s, c in counts.items() if s not in ideal)
        return max(0.005, unexpected / max(1, total))


if _has("braket"):
    register_backend("aws.braket.quera", AWSBraketQuEraBackend, overwrite=True)
    register_backend("aws.braket.ionq", AWSBraketIonQBackend, overwrite=True)
