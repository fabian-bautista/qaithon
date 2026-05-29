"""AWS Braket SV1 simulator backend (cloud-hosted state-vector simulator).

SV1 is Amazon Braket's hosted state-vector simulator. It is a fully
managed service: no QPU, just a powerful classical simulator on AWS
infrastructure. Free Tier allocates ~1 hour of SV1 time per month, which
is the right fit for ``mode="calibrate"`` (one small circuit per forward).
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

__all__ = ["AWSBraketSV1Backend"]

logger = get_logger(__name__)


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


_SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"


class AWSBraketSV1Backend(RealHardwareBackendBase):
    """AWS Braket SV1 simulator with real cloud calibration on demand.

    Args:
        mode: ``"profile"`` | ``"calibrate"`` | ``"execute"``.
        shots: Number of shots per calibration circuit. Default 100.
        device_arn: Braket device ARN. Defaults to SV1.
    """

    profile: BackendProfile = BackendProfile(
        name="aws.braket.sv1",
        kind="quantum",
        energy_pj_per_mac=0.5,  # Classical cloud simulator.
        latency_us_per_op=1_000_000.0,  # ~1 s per task including AWS overhead.
        queue_us=5_000_000.0,  # ~5 s task wait.
        supports_autograd=False,
        supports_batching=False,
        max_dim=2**34,  # SV1 spec: up to 34 qubits.
        notes=(
            "AWS Braket SV1 simulator (hosted state-vector). Free Tier "
            "covers ~1h of SV1 per month. mode='calibrate' uses ~100 ms "
            "per call."
        ),
    )

    def __init__(
        self,
        mode: BackendMode = "profile",
        shots: int = 100,
        device_arn: str = _SV1_ARN,
    ) -> None:
        super().__init__(mode=mode)
        if shots < 1:
            raise ValueError(f"shots must be positive, got {shots}.")
        self._shots = shots
        self._device_arn = device_arn
        self._device = None  # lazy

    def health_check(self):  # type: ignore[no-untyped-def]
        """Ask AWS Braket whether the configured device is ONLINE."""
        from qaithon.backends.base import HealthStatus
        import time as _t

        if not self.is_available():
            return HealthStatus(
                backend="aws.braket.sv1",
                online=False,
                message="braket SDK or AWS credentials missing",
            )
        t0 = _t.perf_counter()
        try:
            device = self._get_device()
            status_str = getattr(device, "status", "UNKNOWN")
            online = str(status_str).upper() == "ONLINE"
            return HealthStatus(
                backend="aws.braket.sv1",
                online=online,
                message=f"AWS Braket SV1 status={status_str}",
                latency_ms=(_t.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(
                backend="aws.braket.sv1",
                online=False,
                message=f"{type(exc).__name__}: {exc}",
                latency_ms=(_t.perf_counter() - t0) * 1000.0,
            )

    def is_available(self) -> bool:
        if not _has("braket"):
            return False
        access_id, secret, _region = get_aws_credentials()
        return bool(access_id and secret)

    def _get_device(self):  # type: ignore[no-untyped-def]
        if self._device is not None:
            return self._device
        from braket.aws import AwsDevice, AwsSession
        import boto3

        access_id, secret, region = get_aws_credentials()
        boto_session = boto3.Session(
            aws_access_key_id=access_id,
            aws_secret_access_key=secret,
            region_name=region,
        )
        aws_session = AwsSession(boto_session=boto_session)
        self._device = AwsDevice(self._device_arn, aws_session=aws_session)
        return self._device

    def _calibrate_once(self) -> float:
        """Run a 3-qubit GHZ on SV1, measure noise (should be ~0 for SV1)."""
        from braket.circuits import Circuit

        circuit = Circuit().h(0).cnot(0, 1).cnot(1, 2)
        device = self._get_device()

        t0 = time.perf_counter()
        task = device.run(circuit, shots=self._shots)
        result = task.result()
        elapsed_us = (time.perf_counter() - t0) * 1e6
        self._last_circuit_latency_us = elapsed_us

        counts = result.measurement_counts
        total = sum(counts.values())
        ideal_strings = {"000", "111"}
        unexpected = sum(c for s, c in counts.items() if s not in ideal_strings)
        noise_fraction = unexpected / max(1, total)
        # SV1 is noiseless by design, so this should be near zero. Use a
        # small floor so the noise injection is not exactly zero.
        return max(0.001, noise_fraction)


if _has("braket"):
    register_backend("aws.braket.sv1", AWSBraketSV1Backend, overwrite=True)
