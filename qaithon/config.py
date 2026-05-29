"""Loading of local credentials from ``.env`` for backends that need them.

Qaithon never hardcodes credentials. Each backend that needs an API key
reads it from the environment at runtime. To keep developer ergonomics
sane, this module loads a project-local ``.env`` file into the environment
the first time it's imported, then exposes typed accessors so the rest
of the code never deals with raw ``os.environ`` strings.

The ``.env`` file is **never** committed (it is in ``.gitignore``). The
companion ``.env.example`` documents every variable Qaithon understands.

Loading rules
-------------

* If ``python-dotenv`` is installed, it is used (most flexible).
* Otherwise a minimal pure-Python parser handles the canonical
  ``KEY=value`` lines so the file still works without an extra dep.
* Existing environment variables are NOT overridden — explicit shell
  exports always win.

Search order for ``.env``:

1. Path in ``QAITHON_ENV_FILE`` environment variable, if set.
2. ``.env`` in the current working directory.
3. ``.env`` in the project root (the directory that contains ``pyproject.toml``).
4. ``.env`` next to this file's package (defensive fallback).
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

from qaithon._logging import get_logger

if TYPE_CHECKING:
    pass

__all__ = [
    "get_aws_credentials",
    "get_ibm_quantum_credentials",
    "get_quandela_credentials",
    "load_env",
]

logger = get_logger(__name__)

_LOADED = False


def _candidate_env_files() -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    explicit = os.environ.get("QAITHON_ENV_FILE")
    if explicit:
        candidates.append(pathlib.Path(explicit))
    candidates.append(pathlib.Path.cwd() / ".env")
    # Walk up from this file to find the project root.
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            candidates.append(parent / ".env")
            break
    candidates.append(here.parent / ".env")
    return candidates


def _parse_env_file_fallback(path: pathlib.Path) -> dict[str, str]:
    """Minimal KEY=value parser. Lines starting with ``#`` are comments."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                out[key] = value
    except OSError as exc:
        logger.debug("Could not read %s: %s", path, exc)
    return out


def load_env(force: bool = False) -> bool:
    """Load credentials from a ``.env`` file into the environment.

    Args:
        force: If ``True``, reload even if a previous call already happened.

    Returns:
        ``True`` if a file was found and loaded, ``False`` if no file was
        found in any of the candidate paths.
    """
    global _LOADED
    if _LOADED and not force:
        return True

    candidates = _candidate_env_files()
    found_path: pathlib.Path | None = None
    for candidate in candidates:
        if candidate.is_file():
            found_path = candidate
            break

    if found_path is None:
        logger.debug("No .env file found in %d candidate paths.", len(candidates))
        return False

    # Prefer python-dotenv when available (handles edge cases like quoted
    # multiline values, escape sequences, etc.).
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(found_path, override=False)
    except ImportError:
        values = _parse_env_file_fallback(found_path)
        for key, value in values.items():
            if key not in os.environ:  # explicit shell exports win
                os.environ[key] = value

    logger.debug("Loaded .env from %s", found_path)
    _LOADED = True
    return True


# Eager load when the module is imported so backends downstream see the values.
load_env()


# ---------------------------------------------------------------------------
# Typed accessors per backend
# ---------------------------------------------------------------------------
def get_ibm_quantum_credentials() -> tuple[str | None, str, str | None]:
    """Return ``(token, channel, instance)`` for the IBM Quantum backend.

    Reads:

    * ``IBM_QUANTUM_TOKEN`` — required.
    * ``IBM_QUANTUM_CHANNEL`` — defaults to ``"ibm_quantum_platform"``.
    * ``IBM_QUANTUM_INSTANCE`` — optional CRN; ``None`` triggers
      auto-detection of the Open-plan instance.
    """
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    channel = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")
    return token, channel, instance


def get_aws_credentials() -> tuple[str | None, str | None, str]:
    """Return ``(access_key_id, secret_access_key, region)`` for AWS Braket.

    Reads ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and
    ``AWS_DEFAULT_REGION`` (defaults to ``"us-east-1"``).
    """
    return (
        os.environ.get("AWS_ACCESS_KEY_ID"),
        os.environ.get("AWS_SECRET_ACCESS_KEY"),
        os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def get_quandela_credentials() -> str | None:
    """Return the Quandela Cloud token or ``None`` if not configured."""
    return os.environ.get("QUANDELA_TOKEN")


def get_huggingface_token() -> str | None:
    """Return the HuggingFace Hub token or ``None`` if not configured.

    Reads either ``HF_TOKEN`` (Qaithon convention) or ``HUGGING_FACE_HUB_TOKEN``
    (the variable ``huggingface_hub`` reads natively).
    """
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# ---------------------------------------------------------------------------
# SDK-style setters
# ---------------------------------------------------------------------------
def set_ibm_token(
    token: str,
    *,
    channel: str = "ibm_quantum_platform",
    instance: str | None = None,
) -> None:
    """Set the IBM Quantum credentials programmatically.

    Args:
        token: 44-character API key from IBM Cloud IAM.
        channel: Authentication channel. Defaults to the modern
            ``"ibm_quantum_platform"``.
        instance: Optional CRN of a specific instance. ``None`` lets
            the runtime auto-detect the Open-plan instance.
    """
    if not token or not isinstance(token, str):
        raise ValueError("token must be a non-empty string.")
    os.environ["IBM_QUANTUM_TOKEN"] = token
    os.environ["IBM_QUANTUM_CHANNEL"] = channel
    if instance is not None:
        os.environ["IBM_QUANTUM_INSTANCE"] = instance


def set_aws_credentials(
    access_key_id: str,
    secret_access_key: str,
    *,
    region: str = "us-east-1",
) -> None:
    """Set AWS Braket credentials programmatically."""
    if not access_key_id or not secret_access_key:
        raise ValueError("access_key_id and secret_access_key must both be set.")
    os.environ["AWS_ACCESS_KEY_ID"] = access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_access_key
    os.environ["AWS_DEFAULT_REGION"] = region


def set_quandela_token(token: str) -> None:
    """Set Quandela Cloud credentials programmatically."""
    if not token or not isinstance(token, str):
        raise ValueError("token must be a non-empty string.")
    os.environ["QUANDELA_TOKEN"] = token


def set_huggingface_token(token: str) -> None:
    """Set HuggingFace Hub credentials programmatically."""
    if not token or not isinstance(token, str):
        raise ValueError("token must be a non-empty string.")
    os.environ["HF_TOKEN"] = token
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def configure(
    *,
    ibm_token: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region: str = "us-east-1",
    quandela_token: str | None = None,
    huggingface_token: str | None = None,
    ibm_channel: str = "ibm_quantum_platform",
    ibm_instance: str | None = None,
) -> None:
    """Configure every cloud provider in one call.

    Pass only what you have. Existing credentials are NOT cleared by
    passing ``None`` — this is purely additive.
    """
    if ibm_token is not None:
        set_ibm_token(ibm_token, channel=ibm_channel, instance=ibm_instance)
    if aws_access_key_id and aws_secret_access_key:
        set_aws_credentials(aws_access_key_id, aws_secret_access_key, region=aws_region)
    if quandela_token is not None:
        set_quandela_token(quandela_token)
    if huggingface_token is not None:
        set_huggingface_token(huggingface_token)


def status() -> dict[str, bool]:
    """Return per-provider booleans of which credentials are configured.

    Useful for diagnostic UIs and automated health checks. Does NOT
    expose the actual values.
    """
    ibm_token, _, _ = get_ibm_quantum_credentials()
    aws_id, aws_secret, _ = get_aws_credentials()
    return {
        "ibm": bool(ibm_token),
        "aws": bool(aws_id and aws_secret),
        "quandela": bool(get_quandela_credentials()),
        "huggingface": bool(get_huggingface_token()),
    }
