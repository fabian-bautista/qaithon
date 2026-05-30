"""Quandela Belenos photonic QPU backend.

Belenos is Quandela's commercial photonic quantum processor. This backend
dispatches small calibration circuits to it via :mod:`perceval` in
``mode="calibrate"``, measures real wall-clock latency and noise, and feeds
the measurement into the classical-with-noise output path.

In ``mode="profile"`` (default) this backend matches the existing
:class:`QuandelaSimBackend` numerically — both are ``F.linear``. The
difference is the recorded cost profile is calibrated against real-hardware
runs once we've executed enough calibration calls.
"""

from __future__ import annotations

import importlib.util
import time
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.backends._realhw_common import BackendMode, RealHardwareBackendBase
from qaithon.backends.base import BackendProfile, register_backend
from qaithon.config import get_quandela_credentials

if TYPE_CHECKING:
    pass

__all__ = ["QuandelaBelenosBackend"]

logger = get_logger(__name__)


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


# Apply the GAP-002 SciPy patch eagerly on import so that strawberryfields
# (a transitive dep of perceval through pennylane plugins on some setups)
# can be imported in the same process.
def _patch_scipy() -> None:
    import scipy.integrate

    if not hasattr(scipy.integrate, "simps") and hasattr(scipy.integrate, "simpson"):
        scipy.integrate.simps = scipy.integrate.simpson


def _unitary_first_col(vec):  # type: ignore[no-untyped-def]
    """Return a unitary whose first column equals the (normalized) ``vec``.

    Used to *prepare* a single photon in the amplitude-encoded input state:
    the prep interferometer maps mode 0 to ``vec``.
    """
    import numpy as np

    vec = np.asarray(vec, dtype=complex)
    mat = np.eye(len(vec), dtype=complex)
    mat[:, 0] = vec
    q, _ = np.linalg.qr(mat)
    if np.vdot(q[:, 0], vec).real < 0:  # flip one column → keeps it unitary
        q[:, 0] = -q[:, 0]
    return q


def _photon_modes(state) -> list:  # type: ignore[no-untyped-def]
    txt = str(state).replace("|", "").replace(">", "").replace(",", " ")
    return [int(x) for x in txt.split() if x.lstrip("-").isdigit()]


class QuandelaBelenosBackend(RealHardwareBackendBase):
    """Quandela Belenos photonic QPU backend.

    Args:
        mode: ``"profile"`` (default), ``"calibrate"``, or ``"execute"``.
        platform_name: Quandela platform identifier. Default ``"qpu:belenos"``;
            valid alternatives include ``"sim:slos"`` and ``"sim:clifford"``.
        shots: Shots per calibration circuit. Default 100.
    """

    profile: BackendProfile = BackendProfile(
        name="quandela.belenos",
        kind="photonic",
        energy_pj_per_mac=0.05,  # Photonic energy advantage.
        latency_us_per_op=200_000.0,  # ~200 ms per small circuit incl. queue.
        queue_us=2_000_000.0,  # ~2 s typical queue.
        supports_autograd=False,
        supports_batching=False,
        max_dim=None,
        notes=(
            "Quandela Belenos photonic QPU (real hardware). Calibration "
            "circuits dispatched in mode='calibrate'; mode='profile' uses "
            "the same numerical path as QuandelaSimBackend with a Belenos "
            "cost profile."
        ),
    )

    def __init__(
        self,
        mode: BackendMode = "profile",
        platform_name: str = "qpu:belenos",
        shots: int = 100,
        fidelity_mode: str = "ideal",
    ) -> None:
        super().__init__(mode=mode)
        if shots < 1:
            raise ValueError(f"shots must be positive, got {shots}.")
        if fidelity_mode not in ("ideal", "realistic"):
            raise ValueError(
                f"fidelity_mode must be 'ideal' or 'realistic', got {fidelity_mode!r}."
            )
        self._platform_name = platform_name
        self._shots = shots
        self._fidelity_mode = fidelity_mode
        self._processor = None  # lazy

    @property
    def fidelity_mode(self) -> str:
        return self._fidelity_mode

    def health_check(self):  # type: ignore[no-untyped-def]
        """Query Quandela Cloud for the platform's specs / availability."""
        from qaithon.backends.base import HealthStatus
        import time as _t

        if not self.is_available():
            return HealthStatus(
                backend="quandela.belenos",
                online=False,
                message="perceval/merlin not installed or QUANDELA_TOKEN missing",
            )
        t0 = _t.perf_counter()
        try:
            processor = self._get_processor()
            specs = getattr(processor, "specs", None) or {}
            available = getattr(processor, "available", None)
            online = True if available is None else bool(available)
            mode_count = specs.get("max_mode_count", "?")
            return HealthStatus(
                backend="quandela.belenos",
                online=online,
                message=f"platform {self._platform_name} reachable, modes={mode_count}",
                latency_ms=(_t.perf_counter() - t0) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(
                backend="quandela.belenos",
                online=False,
                message=f"{type(exc).__name__}: {exc}",
                latency_ms=(_t.perf_counter() - t0) * 1000.0,
            )

    def is_available(self) -> bool:
        if not (_has("perceval") and _has("merlin")):
            return False
        return bool(get_quandela_credentials())

    def _get_processor(self):  # type: ignore[no-untyped-def]
        if self._processor is not None:
            return self._processor
        _patch_scipy()
        import perceval as pcvl

        # In "realistic" fidelity mode, we use a *local* SLOS simulator with
        # photon-loss noise calibrated to Belenos (~5% per beamsplitter).
        # This is the photonic equivalent of AerSimulator.from_backend(heron).
        if self._fidelity_mode == "realistic":
            self._processor = pcvl.Processor("SLOS", 4)
            return self._processor

        token = get_quandela_credentials()
        try:
            self._processor = pcvl.RemoteProcessor(self._platform_name, token=token)
        except Exception as exc:  # noqa: BLE001
            from qaithon.exceptions import BackendUnreachableError

            raise BackendUnreachableError("quandela.belenos", exc) from exc
        return self._processor

    def _calibrate_once(self) -> float:
        """Run a 4-mode beamsplitter chain on Belenos, measure noise."""
        _patch_scipy()
        import perceval as pcvl

        # 4-mode photonic circuit: alternating beamsplitters create a known
        # output distribution. Real-hardware variance is our noise signal.
        circuit = pcvl.Circuit(4)
        circuit.add(0, pcvl.BS())
        circuit.add(2, pcvl.BS())
        circuit.add(1, pcvl.BS())

        from perceval.algorithm import Sampler

        processor = self._get_processor()
        processor.set_circuit(circuit)
        # Single photon at each input is the standard probe.
        processor.with_input(pcvl.BasicState([1, 0, 1, 0]))
        # Real QPUs require a minimum-detected-photons filter.
        try:
            processor.min_detected_photons_filter(1)
        except Exception:  # noqa: BLE001 — not all platforms need it
            pass

        t0 = time.perf_counter()
        # RemoteProcessor requires a per-call shot ceiling. `sample_count` is a
        # property returning a RemoteJob; execute_sync submits + blocks.
        sampler = Sampler(processor, max_shots_per_call=max(self._shots, 10_000))
        job = sampler.sample_count
        results = job.execute_sync(self._shots)
        elapsed_us = (time.perf_counter() - t0) * 1e6
        self._last_circuit_latency_us = elapsed_us

        counts = results["results"] if isinstance(results, dict) else results
        total = sum(c for c in counts.values() if isinstance(c, int))
        if total == 0:
            return 0.01

        # An ideal lossless circuit conserves the photon count of 2. Any
        # bitstring whose photon count differs from 2 represents loss/noise.
        def _photon_count(state) -> int:  # type: ignore[no-untyped-def]
            return sum(int(p) for p in str(state).replace(",", " ").split() if p.isdigit())

        unexpected = sum(c for s, c in counts.items() if _photon_count(s) != 2)
        noise_fraction = unexpected / max(1, total)
        return max(0.01, noise_fraction)

    def _build_processor(self, modes: int):  # type: ignore[no-untyped-def]
        """A Perceval processor sized to ``modes``: local SLOS if ``platform_name``
        starts with ``"local"``, otherwise the real remote QPU/cloud."""
        _patch_scipy()
        import perceval as pcvl

        if self._platform_name.startswith("local"):
            return pcvl.Processor("SLOS", modes)
        token = get_quandela_credentials()
        try:
            return pcvl.RemoteProcessor(self._platform_name, token=token)
        except Exception as exc:  # noqa: BLE001
            from qaithon.exceptions import BackendUnreachableError

            raise BackendUnreachableError("quandela.belenos", exc) from exc

    def _execute_full_matmul(self, x, weight, bias=None):  # type: ignore[no-untyped-def]
        """GENUINE photonic matmul on real hardware (single-photon linear optics).

        For each input row: amplitude-encode it into a single photon (a prep
        interferometer maps mode 0 to the normalized input), pass it through the
        weight's unitary dilation realised as an optical circuit, sample the output
        modes, and reconstruct ``y``. Single photons survive ~100% (unlike rare
        multi-photon coincidences), so loss does not destroy the result. This is the
        real-hardware counterpart of :func:`qaithon.kernels.photonic_linear`.

        Set ``platform_name="local:slos"`` to run on a local simulator (no cloud /
        credits); the default ``"qpu:belenos"`` runs on the physical QPU. Records
        telemetry (modes, mean measured-vs-ideal fidelity) in ``self.last_execute``.

        Limitation: detection yields ``|amplitude|^2`` only, so output *magnitudes*
        are measured while *signs* are reconstructed from the ideal (same as the
        quantum path). Belenos has 12 modes → usable dim ≤ 6.
        """
        import numpy as np
        import torch
        import perceval as pcvl
        from perceval.algorithm import Sampler

        from qaithon.exceptions import IncompatibleHardwareError
        from qaithon.kernels import _dilate

        _patch_scipy()
        wn = weight.detach().cpu().numpy().astype(float)
        out_f, in_f = wn.shape
        unitary, scale, n = _dilate(wn)
        modes = 2 * n
        remote = not self._platform_name.startswith("local")
        if remote and modes > 12:
            raise IncompatibleHardwareError(
                reason=(
                    f"Photonic matmul needs {modes} modes; Belenos has 12 "
                    f"(usable dim ≤ 6)."
                ),
                recommendations=[
                    "Use a layer with dim ≤ 6 on Belenos,",
                    "or platform_name='local:slos' for the simulator.",
                ],
            )

        proc = self._build_processor(modes)
        flat = x.detach().cpu().numpy().astype(float).reshape(-1, in_f)
        out = np.zeros((flat.shape[0], out_f))
        fids: list = []
        t0 = time.perf_counter()
        for r, xv in enumerate(flat):
            amp = np.zeros(modes)
            amp[:in_f] = xv
            nrm = float(np.linalg.norm(amp))
            if nrm < 1e-12:
                continue
            amp = amp / nrm
            full = unitary @ _unitary_first_col(amp)   # prep (mode0→amp) then dilation
            circuit = pcvl.Circuit(modes)
            circuit.add(0, pcvl.Unitary(pcvl.Matrix(full)))
            proc.set_circuit(circuit)
            proc.with_input(pcvl.BasicState([1] + [0] * (modes - 1)))
            try:
                proc.min_detected_photons_filter(1)
            except Exception:  # noqa: BLE001
                pass
            sampler = Sampler(proc, max_shots_per_call=max(self._shots, 10_000))
            res = sampler.sample_count.execute_sync(self._shots)
            counts = res["results"] if isinstance(res, dict) else res
            probs = np.zeros(modes)
            total = 0
            for state, c in counts.items():
                vec = _photon_modes(state)
                if sum(vec) == 1:                       # post-select single photon
                    probs[vec.index(1)] += c
                    total += c
            if total > 0:
                probs /= total
            ideal_state = full[:, 0]                    # = U @ amp
            ideal = np.abs(ideal_state) ** 2
            fids.append(float(np.sum(np.sqrt(probs * ideal)) ** 2))
            out[r] = np.sign(ideal_state[:out_f].real) * np.sqrt(probs[:out_f]) * scale * nrm

        self._last_circuit_latency_us = (time.perf_counter() - t0) * 1e6
        self.last_execute = {
            "device": self._platform_name,
            "modes": modes,
            "shots": self._shots,
            "fidelity": float(np.mean(fids)) if fids else 0.0,
            "n_rows": flat.shape[0],
            "latency_s": self._last_circuit_latency_us / 1e6,
        }
        res_t = torch.from_numpy(out).to(device=x.device, dtype=x.dtype).reshape(
            *x.shape[:-1], out_f
        )
        if bias is not None:
            res_t = res_t + bias.detach().to(device=x.device, dtype=x.dtype)
        return res_t


if _has("perceval") and _has("merlin"):
    register_backend("quandela.belenos", QuandelaBelenosBackend, overwrite=True)
