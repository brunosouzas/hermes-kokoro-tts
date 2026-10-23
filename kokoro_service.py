"""Install and manage the local Kokoro server (launchd on macOS/Apple Silicon, systemd on Linux).

The server runs outside Hermes' Python, in its own virtual environment under the
service directory, so MLX or PyTorch and their dependencies never enter Hermes' environment.
Everything here shells out to ``uv`` and ``launchctl`` or ``systemctl --user``; nothing runs with sudo.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from xml.sax.saxutils import escape

PLUGIN_ROOT = Path(__file__).resolve().parent
SERVER_SOURCES = (PLUGIN_ROOT / "server" / "kokoro_server.py", PLUGIN_ROOT / "kokoro_voices.py",
                  PLUGIN_ROOT / "kokoro_language.py", PLUGIN_ROOT / "kokoro_loanwords.py")
REQUIREMENTS = PLUGIN_ROOT / "server" / "requirements.lock"
LINUX_REQUIREMENTS = PLUGIN_ROOT / "server" / "requirements-linux.lock"

# Reviewed model pins, one per backend: changing one is a code change, not a setting.
MODEL_REPO = "mlx-community/Kokoro-82M-bf16"  # macOS, mlx-audio
MODEL_REVISION = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
LINUX_MODEL_REPO = "hexgrad/Kokoro-82M"  # Linux, the PyTorch kokoro package on CPU
LINUX_MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
PYTHON_VERSION = "3.11"

DEFAULT_PORT = 8770
DEFAULT_LABEL = "dev.hermes.kokoro-tts"


def _linux() -> bool:
    return sys.platform.startswith("linux")


def _xdg_dir(variable: str, fallback: str) -> Path:
    """An XDG base directory; the spec says to ignore a relative value."""
    value = os.environ.get(variable, "")
    return Path(value) if os.path.isabs(value) else Path.home() / fallback


DEFAULT_SERVICE_DIR = (_xdg_dir("XDG_DATA_HOME", ".local/share") if _linux()
                       else Path.home() / "Library" / "Application Support") / "hermes-kokoro-tts"

Runner = Callable[..., subprocess.CompletedProcess]

# Loopback requests must never follow http_proxy/HTTPS_PROXY: the text being spoken would leave the machine.
LOOPBACK_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_port(value: object) -> int:
    """The configured port, or DEFAULT_PORT when it is missing or not a usable TCP port.
    Shared by the provider and the installer so both always agree."""
    try:
        port = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_PORT
    return port if 1 <= port <= 65535 else DEFAULT_PORT


@dataclass
class ServiceConfig:
    port: int = DEFAULT_PORT
    service_dir: Path = DEFAULT_SERVICE_DIR
    label: str = DEFAULT_LABEL
    default_voice: str = "af_heart"
    prime_voices: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def python(self) -> Path:
        return self.service_dir / ".venv" / "bin" / "python"

    @property
    def hf_home(self) -> Path:
        return self.service_dir / "hf-home"

    @property
    def plist_path(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{self.label}.plist"

    @property
    def unit_name(self) -> str:
        return f"{self.label}.service"

    @property
    def unit_path(self) -> Path:
        return _xdg_dir("XDG_CONFIG_HOME", ".config") / "systemd" / "user" / self.unit_name

    @property
    def service_file(self) -> Path:
        """What install writes: the systemd user unit on Linux, the launchd agent on macOS."""
        return self.unit_path if _linux() else self.plist_path

    @property
    def log_path(self) -> Path:
        return self.service_dir / "logs" / "server.log"


def check_platform() -> None:
    machine = platform.machine()
    if (sys.platform == "darwin" and machine == "arm64") or (_linux() and machine in ("x86_64", "aarch64")):
        return
    raise RuntimeError("hermes-kokoro-tts needs macOS on Apple Silicon (MLX) or Linux on x86_64 or aarch64 "
                       "(PyTorch on CPU).")


def find_uv() -> str:
    for candidate in (shutil.which("uv"), "/opt/homebrew/bin/uv", str(Path.home() / ".local" / "bin" / "uv")):
        if candidate and Path(candidate).is_file():
            return candidate
    hint = "`pipx install uv`, your distribution's package," if _linux() else "`brew install uv`"
    raise RuntimeError(f"uv is required to build the server environment: {hint} "
                       "or see https://docs.astral.sh/uv/getting-started/installation/")


def _server_args(cfg: ServiceConfig, repo: str, revision: str) -> list[str]:
    args = [str(cfg.python), str(cfg.service_dir / "server" / "kokoro_server.py"),
            "--host", "127.0.0.1", "--port", str(cfg.port),
            "--model", repo, "--revision", revision, "--default-voice", cfg.default_voice]
    for voice in cfg.prime_voices:
        args += ["--prime", voice]
    return args


def _server_env(cfg: ServiceConfig) -> dict[str, str]:
    return {"HF_HOME": str(cfg.hf_home), "HF_HUB_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}


def render_plist(cfg: ServiceConfig) -> str:
    args = _server_args(cfg, MODEL_REPO, MODEL_REVISION)
    arg_xml = "\n".join(f"\t\t<string>{escape(a)}</string>" for a in args)
    env = _server_env(cfg)
    env_xml = "\n".join(f"\t\t<key>{escape(k)}</key>\n\t\t<string>{escape(v)}</string>" for k, v in env.items())
    log = escape(str(cfg.log_path))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
\t<key>Label</key>
\t<string>{escape(cfg.label)}</string>
\t<key>ProgramArguments</key>
\t<array>
{arg_xml}
\t</array>
\t<key>EnvironmentVariables</key>
\t<dict>
{env_xml}
\t</dict>
\t<key>RunAtLoad</key>
\t<true/>
\t<key>KeepAlive</key>
\t<true/>
\t<key>ThrottleInterval</key>
\t<integer>30</integer>
\t<key>StandardOutPath</key>
\t<string>{log}</string>
\t<key>StandardErrorPath</key>
\t<string>{log}</string>
</dict>
</plist>
"""


def _unit_word(value: str, exec_arg: bool = False) -> str:
    """One quoted word of a systemd unit file. ``%`` starts a specifier everywhere, ``$`` a variable
    only in Exec arguments (not the command itself); both are escaped so a path is taken literally."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"line break in {value!r}")
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return '"' + (value.replace("$", "$$") if exec_arg else value) + '"'


def render_unit(cfg: ServiceConfig) -> str:
    """systemd user unit with the same command, environment and restart policy as the launchd agent."""
    command, *args = _server_args(cfg, LINUX_MODEL_REPO, LINUX_MODEL_REVISION)
    exec_start = " ".join([_unit_word(command), *(_unit_word(a, exec_arg=True) for a in args)])
    env = "\n".join(f"Environment={_unit_word(f'{k}={v}')}" for k, v in _server_env(cfg).items())
    log = str(cfg.log_path).replace("%", "%%")  # inside service_dir, already checked by _unit_word
    return f"""[Unit]
Description=Hermes Kokoro text-to-speech server (loopback only)

[Service]
ExecStart={exec_start}
{env}
Restart=always
RestartSec=30
StandardOutput=append:{log}
StandardError=append:{log}

[Install]
WantedBy=default.target
"""


def _clean_env() -> dict[str, str]:
    """Child environment without the interpreter variables of the Hermes process."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(cfg: ServiceConfig, run: Runner = subprocess.run, say: Callable[[str], None] = print) -> None:
    """Build the environment, download the pinned model and (re)start the launchd agent or systemd unit.
    Idempotent."""
    check_platform()
    linux = _linux()
    repo, revision = (LINUX_MODEL_REPO, LINUX_MODEL_REVISION) if linux else (MODEL_REPO, MODEL_REVISION)
    uv = find_uv()
    env = _clean_env()
    server_dir = cfg.service_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    for src in SERVER_SOURCES:
        shutil.copy2(src, server_dir / src.name)

    say("Preparing the Python environment (the first run downloads ~1.1 GB of packages)...")
    if not cfg.python.is_file():
        run([uv, "venv", "--python", PYTHON_VERSION, str(cfg.service_dir / ".venv")], check=True, env=env)
    run([uv, "pip", "sync", "--python", str(cfg.python), str(LINUX_REQUIREMENTS if linux else REQUIREMENTS)],
        check=True, env=env)

    say(f"Downloading {repo} at revision {revision[:8]} (~340 MB, once)...")
    run([str(cfg.python), "-I", str(server_dir / "kokoro_server.py"), "--download",
         "--model", repo, "--revision", revision], check=True,
        env={**env, "HF_HOME": str(cfg.hf_home)})

    if linux:
        cfg.unit_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.unit_path.write_text(render_unit(cfg))
        run(["systemd-analyze", "--user", "verify", str(cfg.unit_path)], check=True, capture_output=True)
        restart(cfg, run)
        run(["systemctl", "--user", "enable", cfg.unit_name], check=True, capture_output=True)
        say(f"Kokoro server installed: {cfg.base_url} (systemd user unit {cfg.unit_name}).")
        return
    cfg.plist_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.plist_path.write_text(render_plist(cfg))
    run(["plutil", "-lint", str(cfg.plist_path)], check=True, capture_output=True)
    restart(cfg, run)
    say(f"Kokoro server installed: {cfg.base_url} (launchd label {cfg.label}).")


def restart(cfg: ServiceConfig, run: Runner = subprocess.run) -> None:
    if _linux():
        run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True)
        run(["systemctl", "--user", "restart", cfg.unit_name], check=True, capture_output=True)
        return
    target = f"{_domain()}/{cfg.label}"
    run(["launchctl", "bootout", target], check=False, capture_output=True)
    # bootout is asynchronous; bootstrap fails while the previous job is still loaded.
    for _ in range(20):
        if run(["launchctl", "print", target], check=False, capture_output=True).returncode != 0:
            break
        _sleep(0.5)
    run(["launchctl", "bootstrap", _domain(), str(cfg.plist_path)], check=True, capture_output=True)


def uninstall(cfg: ServiceConfig, purge: bool = False, run: Runner = subprocess.run) -> None:
    if _linux():
        run(["systemctl", "--user", "disable", "--now", cfg.unit_name], check=False, capture_output=True)
        cfg.unit_path.unlink(missing_ok=True)
        run(["systemctl", "--user", "daemon-reload"], check=False, capture_output=True)
    else:
        run(["launchctl", "bootout", f"{_domain()}/{cfg.label}"], check=False, capture_output=True)
        if cfg.plist_path.exists():
            cfg.plist_path.unlink()
    if purge and cfg.service_dir.exists():
        shutil.rmtree(cfg.service_dir)


def health(cfg: ServiceConfig, timeout: float = 2.0) -> Optional[dict]:
    try:
        with LOOPBACK_OPENER.open(f"{cfg.base_url}/health", timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def status(cfg: ServiceConfig, run: Runner = subprocess.run) -> dict:
    if _linux():
        active = run(["systemctl", "--user", "is-active", "--quiet", cfg.unit_name],
                     check=False, capture_output=True).returncode == 0
        return {"systemd_unit": cfg.unit_name, "systemd_active": active, "unit_file": str(cfg.unit_path),
                "service_dir": str(cfg.service_dir), "base_url": cfg.base_url, "log": str(cfg.log_path),
                "health": health(cfg)}
    loaded = run(["launchctl", "print", f"{_domain()}/{cfg.label}"], check=False, capture_output=True).returncode == 0
    return {"launchd_label": cfg.label, "launchd_loaded": loaded, "plist": str(cfg.plist_path),
            "service_dir": str(cfg.service_dir), "base_url": cfg.base_url, "log": str(cfg.log_path),
            "health": health(cfg)}


def _sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)
