"""Service management without touching launchd: every command goes through a recording runner."""

import plistlib
import subprocess
from pathlib import Path

import pytest

from kokoro_tts import kokoro_cli, kokoro_service as svc


class Runner:
    def __init__(self, loaded_checks=1):
        self.calls = []
        self._loaded_checks = loaded_checks

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        code = 0
        if cmd[:2] == ["launchctl", "print"]:
            code = 0 if self._loaded_checks > 0 else 113
            self._loaded_checks -= 1
        return subprocess.CompletedProcess(cmd, code, b"", b"")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(svc.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(svc, "_sleep", lambda s: None)
    return svc.ServiceConfig(port=8799, service_dir=tmp_path / "svc", label="test.kokoro", default_voice="pf_dora")


def test_plist_is_valid_loopback_and_pinned(cfg):
    plist = plistlib.loads(svc.render_plist(cfg).encode())
    args = plist["ProgramArguments"]
    assert plist["Label"] == "test.kokoro" and plist["KeepAlive"] is True
    assert args[0] == str(cfg.python)
    assert args[args.index("--host") + 1] == "127.0.0.1" and args[args.index("--port") + 1] == "8799"
    assert args[args.index("--revision") + 1] == svc.MODEL_REVISION and len(svc.MODEL_REVISION) == 40
    assert args[args.index("--default-voice") + 1] == "pf_dora"
    assert plist["EnvironmentVariables"]["HF_HUB_OFFLINE"] == "1"


def test_install_builds_env_downloads_model_and_starts(cfg, monkeypatch):
    monkeypatch.setattr(svc, "check_platform", lambda: None)
    monkeypatch.setattr(svc, "find_uv", lambda: "/opt/uv")
    run = Runner(loaded_checks=1)
    svc.install(cfg, run=run, say=lambda m: None)
    cmds = [c[:3] for c in run.calls]
    assert ["/opt/uv", "venv", "--python"] in cmds
    assert ["/opt/uv", "pip", "sync"] in cmds
    download = next(c for c in run.calls if c[0] == str(cfg.python))
    assert "-I" in download and "--download" in download
    assert svc.MODEL_REPO in download and svc.MODEL_REVISION in download
    assert run.calls[-1][:2] == ["launchctl", "bootstrap"]
    assert (cfg.service_dir / "server" / "kokoro_server.py").is_file()
    assert (cfg.service_dir / "server" / "kokoro_voices.py").is_file()
    assert cfg.plist_path.is_file()


def test_install_skips_venv_creation_when_present(cfg, monkeypatch):
    monkeypatch.setattr(svc, "check_platform", lambda: None)
    monkeypatch.setattr(svc, "find_uv", lambda: "/opt/uv")
    cfg.python.parent.mkdir(parents=True)
    cfg.python.write_text("")
    run = Runner(loaded_checks=0)
    svc.install(cfg, run=run, say=lambda m: None)
    assert not any(c[1:2] == ["venv"] for c in run.calls)


def test_check_platform_refuses_elsewhere(monkeypatch):
    monkeypatch.setattr(svc.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        svc.check_platform()


def test_uninstall_purge(cfg):
    cfg.service_dir.mkdir(parents=True)
    cfg.plist_path.parent.mkdir(parents=True)
    cfg.plist_path.write_text("x")
    svc.uninstall(cfg, purge=True, run=Runner())
    assert not cfg.plist_path.exists() and not cfg.service_dir.exists()


def test_cli_config_from_settings(tmp_path):
    values = {"port": "8801", "service_dir": str(tmp_path / "s"), "voice": "pf_dora"}
    cfg = kokoro_cli.service_config(lambda k, d=None: values.get(k, d))
    assert (cfg.port, cfg.service_dir, cfg.default_voice) == (8801, tmp_path / "s", "pf_dora")
    default = kokoro_cli.service_config(lambda k, d=None: d)
    assert default.service_dir == svc.DEFAULT_SERVICE_DIR and default.port == 8770
