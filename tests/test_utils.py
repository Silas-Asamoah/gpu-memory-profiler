import os
import platform
from types import SimpleNamespace

from gpumemprof.utils import get_system_info


def test_get_system_info_contains_expected_keys():
    system_info = get_system_info()

    assert "platform" in system_info
    assert "architecture" in system_info
    assert "python_version" in system_info
    assert system_info["platform"]
    assert system_info["architecture"]


def test_get_system_info_falls_back_to_platform_module(monkeypatch):
    dummy_uname = SimpleNamespace(system="TestOS", machine="TestArch")
    monkeypatch.delattr(os, "uname", raising=False)
    monkeypatch.setattr(platform, "uname", lambda: dummy_uname)

    system_info = get_system_info()

    assert system_info["platform"] == "TestOS"
    assert system_info["architecture"] == "TestArch"

