"""Hermes TTS provider that speaks through the local Kokoro server."""

from __future__ import annotations

import json
import os
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agent.tts_provider import TTSProvider

from . import kokoro_language as language
from . import kokoro_voices as voices
from .kokoro_service import DEFAULT_LABEL, LOOPBACK_OPENER, parse_port

SAMPLE_RATE = 24000
CONNECT_TIMEOUT = 0.25
REQUEST_TIMEOUT = 120.0
STREAM_CHUNK = 8192

Setting = Callable[[str, Any], Any]


class KokoroServerError(RuntimeError):
    pass


def _wav(pcm: bytes) -> bytes:
    """Mono int16 PCM at SAMPLE_RATE wrapped in a WAV container."""
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm)
    return buf.getvalue()


class _PCMStream:
    """Iterator over a reply's PCM that owns the eagerly opened first response: ``close()``
    releases it even when the consumer stops (barge-in) before reading a single chunk."""

    def __init__(self, first: Any, chunks: Iterator[bytes]):
        self._first = first
        self._chunks = chunks

    def __iter__(self) -> "_PCMStream":
        return self

    def __next__(self) -> bytes:
        return next(self._chunks)

    def close(self) -> None:
        try:
            self._chunks.close()  # type: ignore[attr-defined]
        finally:
            try:
                self._first.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class KokoroTTSProvider(TTSProvider):
    """``tts.provider: kokoro-mlx``. Settings live under ``plugins.entries.kokoro-tts.settings``."""

    streams_pcm = True
    stream_sample_rate = SAMPLE_RATE

    def __init__(self, setting: Setting):
        self._setting = setting

    # ---- identity ---------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "kokoro-mlx"

    @property
    def display_name(self) -> str:
        return "Kokoro (local, Apple Silicon)"

    # ---- settings ---------------------------------------------------------------------------
    @property
    def port(self) -> int:
        return parse_port(self._setting("port", None))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def default_voice(self) -> Optional[str]:
        voice = self._setting("voice", voices.DEFAULT_VOICE)
        return voice if voices.is_valid_voice_id(voice) else voices.DEFAULT_VOICE

    def _speed(self, speed: Optional[float]) -> float:
        value = speed if speed is not None else self._setting("speed", 1.0)
        try:
            return min(4.0, max(0.25, float(value)))
        except (TypeError, ValueError):
            return 1.0

    def _resolve_voice(self, voice: Optional[str]) -> str:
        # ``voice`` comes from tts.voice when set; the plugin setting is the fallback.
        chosen = voice or self.default_voice()
        if not voices.is_valid_voice_id(chosen):
            raise KokoroServerError(f"tts voice {chosen!r} is not a Kokoro voice id (for example 'af_heart' or "
                                    "'pf_dora'); run `hermes kokoro voices` for the list")
        return chosen

    def _lang(self, voice: str, auto: bool = False) -> str:
        # With per-language voices every run is already in its voice's language; the
        # ``language`` override only applies to the single-voice mode.
        return voices.lang_code_for(voice, None if auto else (self._setting("language", "") or None))

    def language_voices(self) -> dict[str, str]:
        """``settings.voices``: detected language (``pt``, ``en``) -> Kokoro voice id. Invalid
        entries are ignored, so a typo falls back to the default voice instead of failing."""
        raw = self._setting("voices", None)
        if not isinstance(raw, dict):
            return {}
        return {str(lang).strip().lower(): voice for lang, voice in raw.items()
                if str(lang).strip().lower() in language.SUPPORTED and voices.is_valid_voice_id(voice)}

    def _runs(self, text: str, voice: Optional[str]) -> list[tuple[str, str, bool]]:
        """``(text, voice, auto)`` requests for ``text``. An explicit voice (``tts.voice``) or an
        empty ``voices`` map keeps today's single-voice behaviour."""
        mapping = self.language_voices()
        if voice or not mapping:
            return [(text, self._resolve_voice(voice), False)]
        return [(part, chosen, True) for part, chosen in language.plan(text, mapping, self._resolve_voice(None))]

    # ---- availability -------------------------------------------------------------------------
    def is_available(self) -> bool:
        """The server is installed (its launchd agent exists). Pickers call this on every paint, so
        the contract forbids network calls; a stopped server surfaces as a clear synthesize() error."""
        try:
            return self._plist_path().is_file()
        except OSError:
            return False

    def _plist_path(self) -> Path:
        return Path.home() / "Library" / "LaunchAgents" / f"{DEFAULT_LABEL}.plist"

    def _server_reachable(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=CONNECT_TIMEOUT):
                return True
        except OSError:
            return False

    def list_voices(self) -> list[dict[str, Any]]:
        try:
            with LOOPBACK_OPENER.open(f"{self.base_url}/v1/voices", timeout=2.0) as resp:
                ids = json.load(resp).get("voices", [])
        except (urllib.error.URLError, OSError, ValueError):
            return []
        return [voices.describe(v) for v in ids if voices.is_valid_voice_id(v)]

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": "kokoro-82m", "display": "Kokoro-82M (MLX, bf16)"}]

    def default_model(self) -> Optional[str]:
        return "kokoro-82m"

    # ---- synthesis --------------------------------------------------------------------------
    def _post(self, text: str, voice: Optional[str], speed: Optional[float], fmt: str, timeout: float,
              auto: bool = False):
        chosen = self._resolve_voice(voice)
        payload = {"input": text, "voice": chosen, "speed": self._speed(speed), "response_format": fmt,
                   "lang_code": self._lang(chosen, auto)}
        req = urllib.request.Request(f"{self.base_url}/v1/audio/speech", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            return LOOPBACK_OPENER.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                message = json.load(exc).get("error", {}).get("message", "")
            except (ValueError, AttributeError):
                message = ""
            raise KokoroServerError(f"Kokoro server returned {exc.code}: {message or exc.reason}") from None
        except (urllib.error.URLError, OSError) as exc:
            raise KokoroServerError(f"Kokoro server is not reachable at {self.base_url} ({exc}); "
                                    "run `hermes kokoro status`, or `hermes kokoro install` once") from None

    def synthesize(self, text: str, output_path: str, *, voice: Optional[str] = None, model: Optional[str] = None,
                   speed: Optional[float] = None, format: str = "mp3", **extra: Any) -> str:
        # The server renders WAV only; per the provider contract the path carries the real extension,
        # and Hermes converts for platforms that need another container.
        path = Path(output_path)
        if path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")
        runs = self._runs(text, voice)
        if len(runs) == 1:
            part, chosen, auto = runs[0]
            with self._post(part, chosen, speed, "wav", REQUEST_TIMEOUT, auto) as resp:
                data = resp.read()
            if len(data) <= 44:
                raise KokoroServerError("Kokoro server returned no audio")
        else:
            # One request per language run, joined into a single WAV.
            pcm = bytearray()
            for part, chosen, auto in runs:
                with self._post(part, chosen, speed, "pcm", REQUEST_TIMEOUT, auto) as resp:
                    audio = resp.read()
                # A silent run would otherwise produce a "successful" file missing that sentence.
                if not audio:
                    raise KokoroServerError("Kokoro server returned no audio for part of the text")
                pcm += audio
            data = _wav(bytes(pcm))
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return str(path)

    def stream(self, text: str, *, voice: Optional[str] = None, model: Optional[str] = None,
               format: str = "pcm", **extra: Any) -> Iterator[bytes]:
        """Raw int16 mono PCM at 24 kHz, for Hermes' sentence-by-sentence speaker. Thread-safe:
        each call owns its HTTP request; the server serializes inference."""
        if format != "pcm":
            raise NotImplementedError("Kokoro streams raw PCM only; Hermes falls back to synthesize()")
        speed = extra.get("speed")
        runs = self._runs(text, voice)
        # The first request opens now so a dead server fails here, where Hermes expects it; later
        # runs open as the previous one finishes playing out.
        first_part, first_voice, first_auto = runs[0]
        first = self._post(first_part, first_voice, speed, "pcm", REQUEST_TIMEOUT, first_auto)

        def chunks() -> Iterator[bytes]:
            resp = first
            for index in range(len(runs)):
                if index:
                    part, chosen, auto = runs[index]
                    resp = self._post(part, chosen, speed, "pcm", REQUEST_TIMEOUT, auto)
                with resp:
                    while True:
                        block = resp.read(STREAM_CHUNK)
                        if not block:
                            break
                        yield block
        return _PCMStream(first, chunks())

    def warm(self) -> None:
        """Speech was just turned on: synthesize a short phrase in the background so the
        configured voice's language pipeline is loaded before the first reply."""
        mapping = self.language_voices()
        if mapping:
            # Each distinct voice once, in its own language.
            targets = [(v, True) for v in dict.fromkeys([self.default_voice(), *mapping.values()])]
        else:
            # Single-voice mode: warm exactly what synthesis will use, language override included.
            targets = [(None, False)]

        def _prime() -> None:
            for voice, auto in targets:
                try:
                    with self._post(".", voice, None, "pcm", 60.0, auto=auto) as resp:
                        resp.read()
                except Exception:
                    pass
        if self._server_reachable():
            threading.Thread(target=_prime, name="kokoro-warm", daemon=True).start()
