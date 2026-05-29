"""Tests for the SDK-style credential setters in :mod:`qaithon.config`."""

from __future__ import annotations

import os

import pytest

import qaithon
from qaithon import config


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate env vars per test so cross-pollution can't leak credentials in."""
    keys = (
        "IBM_QUANTUM_TOKEN",
        "IBM_QUANTUM_CHANNEL",
        "IBM_QUANTUM_INSTANCE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "QUANDELA_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    )
    for k in keys:
        monkeypatch.delenv(k, raising=False)
    yield


def test_set_ibm_token_writes_env(clean_env):
    config.set_ibm_token("dummy-token-1234", instance="ibm-cloud-instance-x")
    assert os.environ["IBM_QUANTUM_TOKEN"] == "dummy-token-1234"
    assert os.environ["IBM_QUANTUM_CHANNEL"] == "ibm_quantum_platform"
    assert os.environ["IBM_QUANTUM_INSTANCE"] == "ibm-cloud-instance-x"


def test_set_ibm_token_rejects_empty(clean_env):
    with pytest.raises(ValueError):
        config.set_ibm_token("")


def test_set_ibm_token_custom_channel(clean_env):
    config.set_ibm_token("t", channel="ibm_cloud")
    assert os.environ["IBM_QUANTUM_CHANNEL"] == "ibm_cloud"


def test_set_aws_credentials_writes_env(clean_env):
    config.set_aws_credentials("AKIATEST", "secret-x", region="eu-west-1")
    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIATEST"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "secret-x"
    assert os.environ["AWS_DEFAULT_REGION"] == "eu-west-1"


def test_set_aws_credentials_rejects_partial(clean_env):
    with pytest.raises(ValueError):
        config.set_aws_credentials("", "secret")


def test_set_quandela_token_writes_env(clean_env):
    config.set_quandela_token("qcloud-token-xyz")
    assert os.environ["QUANDELA_TOKEN"] == "qcloud-token-xyz"


def test_set_quandela_token_rejects_empty(clean_env):
    with pytest.raises(ValueError):
        config.set_quandela_token("")


def test_set_huggingface_token_writes_both_aliases(clean_env):
    config.set_huggingface_token("hf-token-abc")
    assert os.environ["HF_TOKEN"] == "hf-token-abc"
    # huggingface_hub itself reads HUGGING_FACE_HUB_TOKEN; we mirror.
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == "hf-token-abc"


def test_status_reports_all_false_when_empty(clean_env):
    assert config.status() == {
        "ibm": False,
        "aws": False,
        "quandela": False,
        "huggingface": False,
    }


def test_status_reflects_partial_setup(clean_env):
    config.set_ibm_token("token")
    config.set_huggingface_token("hf")
    s = config.status()
    assert s == {
        "ibm": True,
        "aws": False,
        "quandela": False,
        "huggingface": True,
    }


def test_status_does_not_leak_values(clean_env):
    secret = "super-secret-token-do-not-leak"
    config.set_ibm_token(secret)
    s = config.status()
    # Bool only — never the raw value.
    assert s["ibm"] is True
    assert secret not in str(s)


def test_configure_one_call_setup(clean_env):
    config.configure(
        ibm_token="ibm-x",
        aws_access_key_id="AKIA",
        aws_secret_access_key="sec",
        aws_region="us-west-2",
        quandela_token="q",
        huggingface_token="hf",
    )
    assert os.environ["IBM_QUANTUM_TOKEN"] == "ibm-x"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "sec"
    assert os.environ["AWS_DEFAULT_REGION"] == "us-west-2"
    assert os.environ["QUANDELA_TOKEN"] == "q"
    assert os.environ["HF_TOKEN"] == "hf"
    assert config.status() == {
        "ibm": True, "aws": True, "quandela": True, "huggingface": True,
    }


def test_configure_is_additive(clean_env):
    config.set_ibm_token("preset")
    config.configure(huggingface_token="hf")
    # Existing IBM not cleared.
    assert os.environ["IBM_QUANTUM_TOKEN"] == "preset"
    assert os.environ["HF_TOKEN"] == "hf"


def test_configure_skips_aws_when_only_one_key(clean_env):
    config.configure(aws_access_key_id="just-id")
    # Both must be present to write anything.
    assert "AWS_ACCESS_KEY_ID" not in os.environ
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ


def test_setters_exposed_at_top_level():
    """The whole point of the SDK: ``qaithon.set_X`` must be reachable."""
    assert qaithon.set_ibm_token is config.set_ibm_token
    assert qaithon.set_aws_credentials is config.set_aws_credentials
    assert qaithon.set_quandela_token is config.set_quandela_token
    assert qaithon.set_huggingface_token is config.set_huggingface_token
    assert qaithon.configure is config.configure


def test_getters_match_setters_round_trip(clean_env):
    config.set_ibm_token("tk", channel="my-channel", instance="my-instance")
    token, channel, instance = config.get_ibm_quantum_credentials()
    assert token == "tk"
    assert channel == "my-channel"
    assert instance == "my-instance"

    config.set_aws_credentials("id", "secret", region="ap-south-1")
    aid, asecret, region = config.get_aws_credentials()
    assert aid == "id"
    assert asecret == "secret"
    assert region == "ap-south-1"

    config.set_quandela_token("qq")
    assert config.get_quandela_credentials() == "qq"

    config.set_huggingface_token("hh")
    assert config.get_huggingface_token() == "hh"
