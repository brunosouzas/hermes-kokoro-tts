"""``hermes kokoro ...`` commands: install, status, restart, voices, say, words, uninstall."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from . import kokoro_loanwords as loanwords
from . import kokoro_service as service
from . import kokoro_voices as voices

Setting = Callable[[str, Any], Any]


def service_config(setting: Setting) -> service.ServiceConfig:
    raw_dir = setting("service_dir", "") or ""
    default_voice = setting("voice", "af_heart") or "af_heart"
    if not voices.is_valid_voice_id(default_voice):
        default_voice = "af_heart"
    raw_map = setting("voices", None)
    mapped = raw_map.values() if isinstance(raw_map, dict) else ()
    prime = tuple(dict.fromkeys(v for v in mapped if voices.is_valid_voice_id(v) and v != default_voice))
    return service.ServiceConfig(
        port=service.parse_port(setting("port", None)),
        service_dir=Path(raw_dir).expanduser() if raw_dir else service.DEFAULT_SERVICE_DIR,
        default_voice=default_voice,
        prime_voices=prime,
    )


def setup_parser(subparser) -> None:
    subs = subparser.add_subparsers(dest="kokoro_command")
    subs.add_parser("install", help="Build the server environment, download the model and start it (idempotent)")
    subs.add_parser("status", help="Show the launchd agent or systemd unit and server health")
    subs.add_parser("restart", help="Restart the server")
    subs.add_parser("voices", help="List the voices of the installed model")
    say = subs.add_parser("say", help="Speak a test sentence into a WAV file")
    say.add_argument("text")
    say.add_argument("--voice", default=None)
    rm = subs.add_parser("uninstall", help="Stop the server and remove its launchd agent or systemd unit")
    rm.add_argument("--purge", action="store_true", help="Also delete the environment and the downloaded model")
    words = subs.add_parser("words", help="English words a non-English voice says in English (list, add, remove)")
    words.add_argument("action", nargs="?", choices=("list", "add", "remove"), default="list")
    words.add_argument("entries", nargs="*", help='words or quoted phrases, e.g. pipeline "pull request"')


def update_words(current: Any, action: str, entries: list[str]) -> tuple[list[str], list[str]]:
    """``(dictionary, rejected)`` after ``add``/``remove``: case-insensitive, order kept, no
    duplicates. An addition the synthesizer would ignore (numbers, more than four words, over
    64 characters, past the size limit) is rejected here instead of being saved silently."""
    words = [w for w in (current if isinstance(current, list) else []) if isinstance(w, str) and w.strip()]
    rejected: list[str] = []
    if action == "add":
        known = {w.strip().lower() for w in words}
        for entry in entries:
            clean = " ".join(entry.split())
            if clean.lower() in known:
                continue
            if len(clean) > 64 or loanwords.entry_words(clean) is None or len(words) >= loanwords.MAX_ENTRIES:
                rejected.append(entry)
                continue
            words.append(clean)
            known.add(clean.lower())
    elif action == "remove":
        drop = {" ".join(e.split()).lower() for e in entries}
        words = [w for w in words if w.strip().lower() not in drop]
    return words, rejected


def make_handler(setting: Setting, provider_factory: Callable[[], Any],
                 save: Callable[[str, Any], None] | None = None) -> Callable:
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
            elif cmd == "words":
                current = setting("english_words", None)
                if args.action == "list":
                    words, _ = update_words(current, "list", [])
                    print("\n".join(words) if words else "No English words yet. Add some: "
                          'hermes kokoro words add pipeline "pull request"')
                    return 0
                if not args.entries:
                    print(f"Usage: hermes kokoro words {args.action} WORD [\"A PHRASE\" ...]")
                    return 2
                if save is None:
                    print("error: this Hermes cannot save plugin settings from here; edit "
                          "plugins.entries.kokoro-tts.settings.english_words in config.yaml")
                    return 1
                words, rejected = update_words(current, args.action, args.entries)
                save("english_words", words)
                for entry in rejected:
                    print(f"skipped {entry!r}: entries are 1 to {loanwords.MAX_ENTRY_WORDS} words of letters, "
                          f"at most 64 characters, and the dictionary holds {loanwords.MAX_ENTRIES}")
                print(f"{len(words)} English word(s). Changes apply to the next reply; no restart needed.")
                return 1 if rejected else 0
            else:
                print("Usage: hermes kokoro {install|status|restart|voices|say|words|uninstall}")
                return 2
        except Exception as exc:
            print(f"error: {exc}")
            return 1
        return 0
    return handler
