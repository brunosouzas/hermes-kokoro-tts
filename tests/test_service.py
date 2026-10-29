"""Service management without touching launchd or systemd: every command goes through a recording runner."""

import plistlib
import shlex
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
        if cmd[:2] == ["launchctl", "print"] or cmd[:3] == ["systemctl", "--user", "is-active"]:
            code = 0 if self._loaded_checks > 0 else 113
            self._loaded_checks -= 1
        return subprocess.CompletedProcess(cmd, code, b"", b"")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(svc.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(svc, "_sleep", lambda s: None)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return svc.ServiceConfig(port=8799, service_dir=tmp_path / "svc", label="test.kokoro", default_voice="pf_dora")


@pytest.fixture
def macos(monkeypatch):
    monkeypatch.setattr(svc.sys, "platform", "darwin")


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(svc.sys, "platform", "linux")


def test_plist_is_valid_loopback_and_pinned(cfg):
    plist = plistlib.loads(svc.render_plist(cfg).encode())
    args = plist["ProgramArguments"]
    assert plist["Label"] == "test.kokoro" and plist["KeepAlive"] is True
    assert args[0] == str(cfg.launcher) and cfg.launcher.name == "kokoro-tts"
    assert args[args.index("--host") + 1] == "127.0.0.1" and args[args.index("--port") + 1] == "8799"
    assert args[args.index("--revision") + 1] == svc.MODEL_REVISION and len(svc.MODEL_REVISION) == 40
    assert args[args.index("--default-voice") + 1] == "pf_dora"
    assert plist["EnvironmentVariables"]["HF_HUB_OFFLINE"] == "1"


def test_install_builds_env_downloads_model_and_starts(cfg, monkeypatch, macos):
    monkeypatch.setattr(svc, "check_platform", lambda: None)
    monkeypatch.setattr(svc, "find_uv", lambda: "/opt/uv")
    run = Runner(loaded_checks=1)
    svc.install(cfg, run=run, say=lambda m: None)
    cmds = [c[:3] for c in run.calls]
    assert ["/opt/uv", "venv", "--python"] in cmds
    assert ["/opt/uv", "pip", "sync"] in cmds
    assert next(c for c in run.calls if c[1:3] == ["pip", "sync"])[-1] == str(svc.REQUIREMENTS)
    download = next(c for c in run.calls if c[0] == str(cfg.python))
    assert "-I" in download and "--download" in download
    assert svc.MODEL_REPO in download and svc.MODEL_REVISION in download
    assert run.calls[-1][:2] == ["launchctl", "bootstrap"]
    assert (cfg.service_dir / "server" / "kokoro_server.py").is_file()
    assert (cfg.service_dir / "server" / "kokoro_voices.py").is_file()
    assert cfg.plist_path.is_file()
    assert cfg.launcher.is_symlink() and str(cfg.launcher.readlink()) == "python"


def test_install_replaces_a_stale_launcher(cfg, monkeypatch, macos):
    monkeypatch.setattr(svc, "check_platform", lambda: None)
    monkeypatch.setattr(svc, "find_uv", lambda: "/opt/uv")
    cfg.launcher.parent.mkdir(parents=True)
    cfg.launcher.write_text("not a link")
    svc.install(cfg, run=Runner(loaded_checks=1), say=lambda m: None)
    assert cfg.launcher.is_symlink() and str(cfg.launcher.readlink()) == "python"


def test_install_skips_venv_creation_when_present(cfg, monkeypatch):
    monkeypatch.setattr(svc, "check_platform", lambda: None)
    monkeypatch.setattr(svc, "find_uv", lambda: "/opt/uv")
    cfg.python.parent.mkdir(parents=True)
    cfg.python.write_text("")
    run = Runner(loaded_checks=0)
    svc.install(cfg, run=run, say=lambda m: None)
    assert not any(c[1:2] == ["venv"] for c in run.calls)


@pytest.mark.parametrize("system,machine,ok", [
    ("darwin", "arm64", True), ("linux", "x86_64", True), ("linux", "aarch64", True),
    ("darwin", "x86_64", False), ("linux", "armv7l", False), ("win32", "AMD64", False),
])
def test_check_platform(monkeypatch, system, machine, ok):
    monkeypatch.setattr(svc.sys, "platform", system)
    monkeypatch.setattr(svc.platform, "machine", lambda: machine)
    if ok:
        svc.check_platform()
    else:
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            svc.check_platform()


def test_uninstall_purge(cfg, macos):
    cfg.service_dir.mkdir(parents=True)
    cfg.plist_path.parent.mkdir(parents=True)
    cfg.plist_path.write_text("x")
    svc.uninstall(cfg, purge=True, run=Runner())
    assert not cfg.plist_path.exists() and not cfg.service_dir.exists()


def test_plist_primes_extra_voices(cfg):
    cfg.prime_voices = ("bf_emma",)
    args = plistlib.loads(svc.render_plist(cfg).encode())["ProgramArguments"]
    assert args[args.index("--prime") + 1] == "bf_emma"


def test_cli_config_primes_mapped_voices_once():
    values = {"voice": "pf_dora", "voices": {"pt": "pf_dora", "en": "bf_emma", "x": "../bad"}}
    cfg = kokoro_cli.service_config(lambda k, d=None: values.get(k, d))
    assert cfg.default_voice == "pf_dora" and cfg.prime_voices == ("bf_emma",)


def test_cli_config_primes_only_voices_a_sentence_can_reach():
    values = {"voice": "af_heart", "voices": {"it": "im_nicola", "es": "ef_dora"}}
    cfg = kokoro_cli.service_config(lambda k, d=None: values.get(k, d))
    assert cfg.prime_voices == ("im_nicola",)


def test_install_warns_about_unusable_voice_map_entries(monkeypatch, capsys):
    values = {"voices": {"pt": "pf_dora", "es": "ef_dora"}}
    monkeypatch.setattr(svc, "install", lambda cfg: None)
    handler = kokoro_cli.make_handler(lambda k, d=None: values.get(k, d), provider_factory=lambda: None)
    assert handler(type("Args", (), {"kokoro_command": "install"})()) == 0
    out = capsys.readouterr().out
    assert "warning: voices.es" in out and "ef_dora" in out


def test_cli_config_from_settings(tmp_path):
    values = {"port": "8801", "service_dir": str(tmp_path / "s"), "voice": "pf_dora"}
    cfg = kokoro_cli.service_config(lambda k, d=None: values.get(k, d))
    assert (cfg.port, cfg.service_dir, cfg.default_voice) == (8801, tmp_path / "s", "pf_dora")
    default = kokoro_cli.service_config(lambda k, d=None: d)
    assert default.service_dir == svc.DEFAULT_SERVICE_DIR and default.port == 8770


def parse_unit(text):
    lines = [line.split("=", 1) for line in text.splitlines() if "=" in line]
    return {k: [v for key, v in lines if key == k] for k, _ in lines}


def test_unit_is_loopback_pinned_offline_and_restarts(cfg):
    cfg.prime_voices = ("bf_emma",)
    unit = parse_unit(svc.render_unit(cfg))
    args = shlex.split(unit["ExecStart"][0])
    assert args[:2] == [str(cfg.launcher), str(cfg.service_dir / "server" / "kokoro_server.py")]
    assert args[args.index("--host") + 1] == "127.0.0.1" and args[args.index("--port") + 1] == "8799"
    assert args[args.index("--model") + 1] == svc.LINUX_MODEL_REPO == "hexgrad/Kokoro-82M"
    assert args[args.index("--revision") + 1] == svc.LINUX_MODEL_REVISION and len(svc.LINUX_MODEL_REVISION) == 40
    assert args[args.index("--default-voice") + 1] == "pf_dora" and args[args.index("--prime") + 1] == "bf_emma"
    assert '"HF_HUB_OFFLINE=1"' in unit["Environment"] and f'"HF_HOME={cfg.hf_home}"' in unit["Environment"]
    assert unit["Restart"] == ["always"] and unit["WantedBy"] == ["default.target"]
    assert unit["StandardOutput"] == [f"append:{cfg.log_path}"]


def test_unit_takes_odd_paths_literally(cfg):
    cfg.service_dir = Path('/srv/a b%c$d"e')
    unit = parse_unit(svc.render_unit(cfg))
    exec_start = unit["ExecStart"][0]
    # systemd expands % everywhere and $ in arguments, but not in the command itself.
    assert exec_start.startswith('"/srv/a b%%c$d\\"e/.venv/bin/kokoro-tts" "/srv/a b%%c$$d\\"e/server/kokoro_server.py"')
    assert '"HF_HOME=/srv/a b%%c$d\\"e/hf-home"' in unit["Environment"]
    cfg.service_dir = Path("/srv/a\nExecStartPre=/bin/evil")
    with pytest.raises(ValueError, match="line break"):
        svc.render_unit(cfg)


def test_install_on_linux_uses_the_linux_lock_pin_and_systemd(cfg, monkeypatch, linux):
    monkeypatch.setattr(svc, "check_platform", lambda: None)
    monkeypatch.setattr(svc, "find_uv", lambda: "/opt/uv")
    run = Runner()
    svc.install(cfg, run=run, say=lambda m: None)
    assert next(c for c in run.calls if c[1:3] == ["pip", "sync"])[-1] == str(svc.LINUX_REQUIREMENTS)
    download = next(c for c in run.calls if c[0] == str(cfg.python))
    assert svc.LINUX_MODEL_REPO in download and svc.LINUX_MODEL_REVISION in download
    assert run.calls[-4:] == [["systemd-analyze", "--user", "verify", str(cfg.unit_path)],
                              ["systemctl", "--user", "daemon-reload"],
                              ["systemctl", "--user", "restart", "test.kokoro.service"],
                              ["systemctl", "--user", "enable", "test.kokoro.service"]]
    assert cfg.unit_path == cfg.service_file and cfg.unit_path.is_file() and not cfg.plist_path.exists()
    assert not any(c[0] == "launchctl" for c in run.calls)


def test_unit_lives_in_the_xdg_config_dir(cfg, monkeypatch, tmp_path, linux):
    assert cfg.unit_path == tmp_path / ".config" / "systemd" / "user" / "test.kokoro.service"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "conf"))
    assert cfg.service_file == tmp_path / "conf" / "systemd" / "user" / "test.kokoro.service"
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/ignored")
    assert cfg.unit_path == tmp_path / ".config" / "systemd" / "user" / "test.kokoro.service"


def test_linux_restart_status_and_uninstall(cfg, linux):
    run = Runner(loaded_checks=1)
    svc.restart(cfg, run=run)
    assert run.calls == [["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "restart", "test.kokoro.service"]]
    state = svc.status(cfg, run=run)
    assert state["systemd_active"] is True and state["unit_file"] == str(cfg.unit_path)
    assert svc.status(cfg, run=run)["systemd_active"] is False  # is-active now answers 113
    cfg.service_dir.mkdir(parents=True)
    cfg.unit_path.parent.mkdir(parents=True)
    cfg.unit_path.write_text("x")
    run = Runner()
    svc.uninstall(cfg, purge=True, run=run)
    assert run.calls == [["systemctl", "--user", "disable", "--now", "test.kokoro.service"],
                         ["systemctl", "--user", "daemon-reload"]]
    assert not cfg.unit_path.exists() and not cfg.service_dir.exists()


def test_cli_shows_why_a_service_command_failed(monkeypatch, capsys):
    import types

    def fail(cfg):
        raise subprocess.CalledProcessError(1, ["systemd-analyze", "--user", "verify", "x"],
                                            stderr=b"x.service: Failed to create directory: Permission denied\n")
    monkeypatch.setattr(svc, "install", fail)
    handler = kokoro_cli.make_handler(lambda k, d=None: d, lambda: None)
    assert handler(types.SimpleNamespace(kokoro_command="install")) == 1
    out = capsys.readouterr().out
    assert "returned non-zero exit status 1" in out and "Permission denied" in out
