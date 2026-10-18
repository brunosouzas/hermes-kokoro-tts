"""``hermes kokoro ...`` commands: install, status, voices, say, uninstall."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from . import kokoro_service as service

Setting = Callable[[str, Any], Any]


def service_config(setting: Setting) -> service.ServiceConfig:
    raw_dir = setting("service_dir", "") or ""
    return service.ServiceConfig(
        port=service.parse_port(setting("port", None)),
        service_dir=Path(raw_dir).expanduser() if raw_dir else service.DEFAULT_SERVICE_DIR,
        default_voice=setting("voice", "af_heart") or "af_heart",
    )


def setup_parser(subparser) -> None:
    subs = subparser.add_subparsers(dest="kokoro_command")
    subs.add_parser("install", help="Build the server environment, download the model and start it (idempotent)")
    subs.add_parser("status", help="Show the launchd agent and server health")
    subs.add_parser("restart", help="Restart the server")
    subs.add_parser("voices", help="List the voices of the installed model")
    say = subs.add_parser("say", help="Speak a test sentence into a WAV file")
    say.add_argument("text")
    say.add_argument("--voice", default=None)
    rm = subs.add_parser("uninstall", help="Stop the server and remove its launchd agent")
    rm.add_argument("--purge", action="store_true", help="Also delete the environment and the downloaded model")


def make_handler(setting: Setting, provider_factory: Callable[[], Any]) -> Callable:
    def handler(args) -> int:
        cfg = service_config(setting)
        cmd = getattr(args, "kokoro_command", None)
        try:
            if cmd == "install":
                raw_port = setting("port", None)
                if raw_port not in (None, "") and str(raw_port) != str(cfg.port):
                    print(f"warning: port setting {raw_port!r} is not a usable TCP port; using {cfg.port}")
                service.install(cfg)
                print("Next: hermes config set tts.provider kokoro-mlx")
            elif cmd == "status":
                print(json.dumps(service.status(cfg), indent=2))
            elif cmd == "restart":
                service.restart(cfg)
                print(f"Restarted {cfg.label}.")
            elif cmd == "voices":
                entries = provider_factory().list_voices()
                if not entries:
                    print(f"No voices: is the server running? Try `hermes kokoro status` ({cfg.base_url}).")
                    return 1
                for entry in entries:
                    print(f"{entry['id']:<16} {entry['display']}")
            elif cmd == "say":
                out = Path(tempfile.gettempdir()) / "hermes-kokoro-say.wav"
                print(provider_factory().synthesize(args.text, str(out), voice=args.voice))
            elif cmd == "uninstall":
                service.uninstall(cfg, purge=args.purge)
                print("Removed." + (" Environment and model deleted." if args.purge else ""))
            else:
                print("Usage: hermes kokoro {install|status|restart|voices|say|uninstall}")
                return 2
        except Exception as exc:
            print(f"error: {exc}")
            return 1
        return 0
    return handler
