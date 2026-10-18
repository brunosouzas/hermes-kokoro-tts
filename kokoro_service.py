"""Install and manage the local Kokoro server (macOS on Apple Silicon, launchd).

The server runs outside Hermes' Python, in its own virtual environment under the
service directory, so MLX and its dependencies never enter Hermes' environment.
Everything here shells out to ``uv`` and ``launchctl``; nothing runs with sudo.
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
SERVER_SOURCES = (PLUGIN_ROOT / "server" / "kokoro_server.py", PLUGIN_ROOT / "kokoro_voices.py")
REQUIREMENTS = PLUGIN_ROOT / "server" / "requirements.lock"

# Reviewed model pin: changing it is a code change, not a setting.
MODEL_REPO = "mlx-community/Kokoro-82M-bf16"
MODEL_REVISION = "a71e4d38b236d968966a2002c4c895dbd12b1c3c"
PYTHON_VERSION = "3.11"

DEFAULT_PORT = 8770
DEFAULT_LABEL = "dev.hermes.kokoro-tts"
DEFAULT_SERVICE_DIR = Path.home() / "Library" / "Application Support" / "hermes-kokoro-tts"

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
    def log_path(self) -> Path:
        return self.service_dir / "logs" / "server.log"


def check_platform() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("hermes-kokoro-tts runs Kokoro on MLX, which needs macOS on Apple Silicon (arm64).")


def find_uv() -> str:
    for candidate in (shutil.which("uv"), "/opt/homebrew/bin/uv", str(Path.home() / ".local" / "bin" / "uv")):
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("uv is required to build the server environment: `brew install uv` "
                       "or see https://docs.astral.sh/uv/getting-started/installation/")


def render_plist(cfg: ServiceConfig) -> str:
    args = [str(cfg.python), str(cfg.service_dir / "server" / "kokoro_server.py"),
            "--host", "127.0.0.1", "--port", str(cfg.port),
            "--model", MODEL_REPO, "--revision", MODEL_REVISION, "--default-voice", cfg.default_voice]
    arg_xml = "\n".join(f"\t\t<string>{escape(a)}</string>" for a in args)
    env = {"HF_HOME": str(cfg.hf_home), "HF_HUB_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
           "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
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


def _clean_env() -> dict[str, str]:
    """Child environment without the interpreter variables of the Hermes process."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _domain() -> str:
    return f"gui/{os.getuid()}"


def install(cfg: ServiceConfig, run: Runner = subprocess.run, say: Callable[[str], None] = print) -> None:
    """Build the environment, download the pinned model and (re)start the launchd agent. Idempotent."""
    check_platform()
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
    run([uv, "pip", "sync", "--python", str(cfg.python), str(REQUIREMENTS)], check=True, env=env)

    say(f"Downloading {MODEL_REPO} at revision {MODEL_REVISION[:8]} (~340 MB, once)...")
    run([str(cfg.python), "-I", str(server_dir / "kokoro_server.py"), "--download",
         "--model", MODEL_REPO, "--revision", MODEL_REVISION], check=True,
        env={**env, "HF_HOME": str(cfg.hf_home)})

    cfg.plist_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.plist_path.write_text(render_plist(cfg))
    run(["plutil", "-lint", str(cfg.plist_path)], check=True, capture_output=True)
    restart(cfg, run)
    say(f"Kokoro server installed: {cfg.base_url} (launchd label {cfg.label}).")


def restart(cfg: ServiceConfig, run: Runner = subprocess.run) -> None:
    target = f"{_domain()}/{cfg.label}"
    run(["launchctl", "bootout", target], check=False, capture_output=True)
    # bootout is asynchronous; bootstrap fails while the previous job is still loaded.
    for _ in range(20):
        if run(["launchctl", "print", target], check=False, capture_output=True).returncode != 0:
            break
        _sleep(0.5)
    run(["launchctl", "bootstrap", _domain(), str(cfg.plist_path)], check=True, capture_output=True)


def uninstall(cfg: ServiceConfig, purge: bool = False, run: Runner = subprocess.run) -> None:
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
    loaded = run(["launchctl", "print", f"{_domain()}/{cfg.label}"], check=False, capture_output=True).returncode == 0
    return {"launchd_label": cfg.label, "launchd_loaded": loaded, "plist": str(cfg.plist_path),
            "service_dir": str(cfg.service_dir), "base_url": cfg.base_url, "log": str(cfg.log_path),
            "health": health(cfg)}


def _sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)
