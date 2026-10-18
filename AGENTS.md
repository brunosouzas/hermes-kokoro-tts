# hermes-kokoro-tts: project context

## Purpose

A public Hermes Agent plugin: local Kokoro text-to-speech on Apple Silicon. The plugin registers the `kokoro` TTS provider and a `hermes kokoro` command; the model runs in a separate, loopback-only server process with its own virtual environment.

## Technology declarations

Stack: Python, Hermes plugin API, MLX (mlx-audio), launchd

- Plugin code (`__init__.py`, `kokoro_*.py`) runs inside Hermes and uses the standard library only. Never add a dependency there; anything heavy belongs to the server.
- Server code (`server/kokoro_server.py`, plus `kokoro_voices.py`, which the installer copies next to it) runs in Python 3.11 from `server/requirements.lock`.
- Tests: `.venv/bin/python -m pytest -q` (see README → Development); validation: `hermes plugins validate .`.

## Rules

- Repository language: English.
- Board: Azure Boards `brunosouzas`, area `jarvis`; branches `ab<card>-<slug>`; pull requests end with `Fixes AB#<card>`; the owner merges.
- The model pin (`MODEL_REPO`, `MODEL_REVISION` in `kokoro_service.py`) changes only in a reviewed pull request, with the reason in its description.
- The server must keep refusing non-loopback addresses, never log request text, and validate voice ids so a voice can never name an arbitrary file.
- No self-updating code (Hermes plugin catalog rule 3).
- Keep README's "What it reads, sends and writes" true after every change.
