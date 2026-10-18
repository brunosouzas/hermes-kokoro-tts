"""Test bootstrap.

The plugin uses relative imports because Hermes loads the repository root as a package. Tests
load it the same way, under the package name ``kokoro_tts``. ``agent.tts_provider`` comes from a
Hermes checkout when one is importable (``HERMES_AGENT_REPO``, default ``~/.hermes/hermes-agent``);
otherwise a minimal stand-in with the same base-class contract is used.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
HERMES_AGENT_REPO = Path(os.environ.get("HERMES_AGENT_REPO") or Path.home() / ".hermes" / "hermes-agent").expanduser()

collect_ignore = ["__init__.py", "kokoro_cli.py", "kokoro_provider.py", "kokoro_service.py", "kokoro_voices.py", "kokoro_language.py",
                  "server"]


def _ensure_tts_provider_base() -> None:
    if HERMES_AGENT_REPO.is_dir() and str(HERMES_AGENT_REPO) not in sys.path:
        sys.path.append(str(HERMES_AGENT_REPO))
    try:
        import agent.tts_provider  # noqa: F401
        return
    except Exception:
        for name in [m for m in sys.modules if m == "agent" or m.startswith("agent.")]:
            del sys.modules[name]
    stub_pkg = types.ModuleType("agent")
    stub_pkg.__path__ = []
    stub = types.ModuleType("agent.tts_provider")

    class TTSProvider:  # same surface the plugin relies on
        streams_pcm = False
        stream_sample_rate = None

        def list_voices(self):
            return []

        def default_voice(self):
            return None

        def warm(self):
            pass

        def release(self):
            pass

    stub.TTSProvider = TTSProvider
    sys.modules["agent"] = stub_pkg
    sys.modules["agent.tts_provider"] = stub


def _load_plugin_package() -> None:
    if "kokoro_tts" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("kokoro_tts", REPO_ROOT / "__init__.py",
                                                  submodule_search_locations=[str(REPO_ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["kokoro_tts"] = module
    spec.loader.exec_module(module)


_ensure_tts_provider_base()
_load_plugin_package()
for path in (str(REPO_ROOT), str(REPO_ROOT / "server")):
    if path not in sys.path:
        sys.path.insert(0, path)
