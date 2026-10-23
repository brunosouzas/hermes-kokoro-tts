# hermes-kokoro-tts: project context

## Purpose

A public Hermes Agent plugin: local Kokoro text-to-speech on Apple Silicon and Linux. The plugin (`kokoro-tts`) registers the `kokoro-mlx` TTS provider (not `kokoro`: Hermes reserves the names of built-in providers, see README → Upgrading from 0.1.0) and a `hermes kokoro` command; the model runs in a separate, loopback-only server process with its own virtual environment.

## Technology declarations

Stack: Python, Hermes plugin API, MLX (mlx-audio) and launchd on macOS, PyTorch (kokoro, CPU) and systemd user units on Linux

- Plugin code (`__init__.py`, `kokoro_*.py`) runs inside Hermes and uses the standard library only. Never add a dependency there; anything heavy belongs to the server.
- Server code runs in Python 3.11 from `server/requirements.lock` (macOS) or `server/requirements-linux.lock` (Linux): `server/kokoro_server.py` plus the shared modules the installer copies next to it (`kokoro_voices.py`, `kokoro_language.py`, `kokoro_loanwords.py`; the list is `SERVER_SOURCES` in `kokoro_service.py`). Those shared modules must stay standard-library only, because the plugin imports them too.
- Tests: `.venv/bin/python -m pytest -q` (see README → Development); validation: `hermes plugins validate .`.

## Rules

- Repository language: English.
- Board: Azure Boards `brunosouzas`, area `jarvis`; branches `ab<card>-<slug>`; pull requests end with `Fixes AB#<card>`; the owner merges.
- The model pins (`MODEL_REPO`, `MODEL_REVISION` for macOS and `LINUX_MODEL_REPO`, `LINUX_MODEL_REVISION` for Linux, in `kokoro_service.py`) change only in a reviewed pull request, with the reason in its description.
- The server must keep refusing non-loopback addresses, never log request text, and validate voice ids so a voice can never name an arbitrary file.
- No self-updating code (Hermes plugin catalog rule 3).
- Keep README's "What it reads, sends and writes" true after every change.
