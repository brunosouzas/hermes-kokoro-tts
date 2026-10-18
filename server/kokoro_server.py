"""Local Kokoro text-to-speech server (Apple Silicon, MLX).

Runs in its own virtual environment, managed by ``hermes kokoro install``. It loads
one pinned Kokoro model once and serves an OpenAI-compatible speech endpoint on the
loopback interface:

    POST /v1/audio/speech   {"input", "voice", "speed", "response_format": "wav"|"pcm", "lang_code"}
    GET  /v1/voices         voice ids available in the pinned model snapshot
    GET  /health            {"ok", "model", "revision", "loaded", "sample_rate"}

Voices are resolved from the pinned snapshot on disk and passed to mlx-audio as a
file path, so the server never touches the network after installation.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import threading
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kokoro_language  # noqa: E402  (these three are copied next to this file by the installer)
import kokoro_loanwords  # noqa: E402
import kokoro_voices  # noqa: E402

SAMPLE_RATE = 24000
MAX_INPUT_CHARS = 10000
MAX_BODY_BYTES = 64 * 1024
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
ENGLISH_CODES = ("a", "b")
MAX_PHONEMES = 510  # Kokoro's limit for one inference call
MAX_ENGLISH_WORDS = kokoro_loanwords.MAX_ENTRIES

log = logging.getLogger("kokoro_server")


class NotReady(Exception):
    """The model has not loaded (yet, or at all)."""


class VoiceNotFound(Exception):
    """The requested voice is not in the installed snapshot."""


class Engine:
    """Owns the loaded model. ``synthesize`` is serialized: MLX inference is not re-entrant."""

    def __init__(self, repo: str, revision: str, loader: Optional[Callable[[], Any]] = None,
                 snapshot: Optional[Path] = None):
        self.repo, self.revision = repo, revision
        self._loader = loader
        self._snapshot = snapshot
        self._model: Any = None
        self._lock = threading.Lock()
        self.error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def snapshot(self) -> Path:
        if self._snapshot is None:
            self._snapshot = download(self.repo, self.revision, local_files_only=True)
        return self._snapshot

    def load(self) -> None:
        try:
            if self._loader is not None:
                self._model = self._loader()
            else:
                import mlx.core as mx
                from mlx_audio.tts.utils import load_model
                self.snapshot()  # fail early if the pinned snapshot is missing
                model = load_model(self.repo, revision=self.revision)
                mx.eval(model.parameters())
                self._model = model
            log.info("model ready: %s@%s", self.repo, self.revision[:8])
        except Exception as exc:  # reported through /health; requests return 503
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model failed to load")

    def voices(self) -> list[str]:
        folder = self.snapshot() / "voices"
        return sorted(p.stem for p in folder.glob("*.safetensors") if kokoro_voices.is_valid_voice_id(p.stem))

    def voice_path(self, voice: str) -> Path:
        if not kokoro_voices.is_valid_voice_id(voice):
            raise ValueError(f"invalid voice id {voice!r}")
        path = self.snapshot() / "voices" / f"{voice}.safetensors"
        if not path.is_file():
            raise VoiceNotFound(f"voice {voice!r} is not in the installed model")
        return path

    def synthesize(self, text: str, voice: str, lang_code: str, speed: float,
                   english_words: Optional[list] = None, english_accent: str = "a") -> bytes:
        """Return mono int16 little-endian PCM at SAMPLE_RATE. ``english_words`` (non-English
        voices only) are pronounced with English phonemes by the same voice."""
        if self._model is None:
            raise NotReady(self.error or "model is still loading")
        voice_file = str(self.voice_path(voice))
        entries = kokoro_loanwords.normalize(english_words or []) if lang_code not in ENGLISH_CODES else []
        with self._lock:
            if not entries:
                return self._generate(text, voice_file, lang_code, speed)
            return self._generate_with_loanwords(text, voice_file, lang_code, speed, entries, english_accent)

    def _generate(self, text: str, voice_file: str, lang_code: str, speed: float) -> bytes:
        pieces = [result.audio for result in
                  self._model.generate(text=text, voice=voice_file, lang_code=lang_code, speed=speed)]
        return _to_pcm(pieces)

    def _g2p(self, lang_code: str) -> Callable[[str], str]:
        pipeline = self._model._get_pipeline(lang_code)

        def phonemize(text: str) -> str:
            result = pipeline.g2p(text)
            return (result[0] if isinstance(result, tuple) else result) or ""
        return phonemize

    def _generate_with_loanwords(self, text: str, voice_file: str, lang_code: str, speed: float,
                                 entries: list, english_accent: str) -> bytes:
        """Sentence by sentence: spliced phonemes where a listed word occurs, the normal path
        everywhere else (and whenever the splice cannot be lined up)."""
        native, english = self._g2p(lang_code), self._g2p(english_accent)
        pipeline = self._model._get_pipeline(lang_code)
        pieces = []
        for sentence in kokoro_language.split_sentences(text) or [text]:
            phonemes = kokoro_loanwords.splice(sentence, entries, native, english)
            if phonemes and len(phonemes) <= MAX_PHONEMES:
                for result in pipeline.generate_from_tokens(phonemes, voice=voice_file, speed=speed,
                                                            model=self._model):
                    pieces.append(result.output.audio)
            else:
                pieces.extend(result.audio for result in self._model.generate(
                    text=sentence, voice=voice_file, lang_code=lang_code, speed=speed))
        return _to_pcm(pieces)


def _to_pcm(pieces: list) -> bytes:
    import numpy as np
    arrays = [np.asarray(p, dtype=np.float32).reshape(-1) for p in pieces]
    if not arrays:
        return b""
    audio = np.clip(np.concatenate(arrays), -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def download(repo: str, revision: str, local_files_only: bool = False) -> Path:
    """Fetch (or, offline, locate) the pinned snapshot.

    Uses the same file patterns as mlx-audio's own loader: huggingface_hub checks an offline
    snapshot for completeness against the patterns it is given, so the install-time download
    and every later offline lookup must agree on them.
    """
    from huggingface_hub import snapshot_download
    from mlx_audio.utils import DEFAULT_ALLOW_PATTERNS
    return Path(snapshot_download(repo, revision=revision, allow_patterns=DEFAULT_ALLOW_PATTERNS,
                                  local_files_only=local_files_only))


# Packages that would otherwise be fetched from the network the first time a language is used.
OFFLINE_PACKAGES = ("en_core_web_sm",)


def check_offline_packages() -> None:
    """Fail installation if a runtime download would be needed (misaki calls spacy.cli.download)."""
    import importlib.util
    missing = [name for name in OFFLINE_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"missing offline packages: {', '.join(missing)}; re-run `hermes kokoro install`")


def pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buf.getvalue()


def make_handler(engine: Engine, default_voice: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "hermes-kokoro-tts"

        def log_message(self, fmt: str, *args: Any) -> None:  # request text is never logged
            log.info("%s %s", self.command, self.path.split("?", 1)[0])

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": {"message": message}})

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._json(HTTPStatus.OK, {"ok": engine.loaded, "loaded": engine.loaded, "error": engine.error,
                                           "model": engine.repo, "revision": engine.revision,
                                           "sample_rate": SAMPLE_RATE})
            elif path == "/v1/voices":
                try:
                    self._json(HTTPStatus.OK, {"voices": engine.voices()})
                except Exception as exc:
                    self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/v1/audio/speech":
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._error(HTTPStatus.BAD_REQUEST, "bad Content-Length")
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body missing or too large")
            try:
                req = json.loads(self.rfile.read(length))
                if not isinstance(req, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                return self._error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}")

            text = req.get("input")
            if not isinstance(text, str) or not text.strip():
                return self._error(HTTPStatus.BAD_REQUEST, "'input' must be non-empty text")
            if len(text) > MAX_INPUT_CHARS:
                return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"'input' exceeds {MAX_INPUT_CHARS} characters")
            voice = req.get("voice") or default_voice
            fmt = req.get("response_format") or "wav"
            lang_override = req.get("lang_code")
            if not isinstance(voice, str):
                return self._error(HTTPStatus.BAD_REQUEST, "'voice' must be a string")
            if not isinstance(fmt, str) or fmt.lower() not in ("wav", "pcm"):
                return self._error(HTTPStatus.BAD_REQUEST, "response_format must be 'wav' or 'pcm'")
            fmt = fmt.lower()
            if lang_override is not None and not isinstance(lang_override, str):
                return self._error(HTTPStatus.BAD_REQUEST, "'lang_code' must be a string")
            english_words = req.get("english_words") or []
            if not isinstance(english_words, list) or len(english_words) > MAX_ENGLISH_WORDS \
                    or not all(isinstance(w, str) and len(w) <= 64 for w in english_words):
                return self._error(HTTPStatus.BAD_REQUEST,
                                   f"'english_words' must be a list of at most {MAX_ENGLISH_WORDS} short strings")
            english_accent = req.get("english_accent") or "a"
            if english_accent not in ENGLISH_CODES:
                return self._error(HTTPStatus.BAD_REQUEST, "'english_accent' must be 'a' (US) or 'b' (UK)")
            try:
                speed = float(req.get("speed") or 1.0)
                if not 0.25 <= speed <= 4.0:
                    raise ValueError
            except (TypeError, ValueError):
                return self._error(HTTPStatus.BAD_REQUEST, "'speed' must be between 0.25 and 4.0")
            # Validate the request first: these messages are ours and never contain the text.
            try:
                lang = kokoro_voices.lang_code_for(voice, lang_override)
                engine.voice_path(voice)
            except ValueError as exc:
                return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except VoiceNotFound as exc:
                return self._error(HTTPStatus.NOT_FOUND, str(exc))
            except NotReady as exc:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            except Exception as exc:
                log.error("voice lookup failed: %s", type(exc).__name__)
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, f"voice lookup failed: {type(exc).__name__}")
            try:
                pcm = engine.synthesize(text, voice, lang, speed, english_words, english_accent)
            except NotReady as exc:
                return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            except Exception as exc:
                # Exception messages and tracebacks can quote the input; log the type only.
                log.error("synthesis failed: %s", type(exc).__name__)
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"synthesis failed: {type(exc).__name__}")

            body = pcm if fmt == "pcm" else pcm_to_wav(pcm)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/pcm; rate=24000" if fmt == "pcm" else "audio/wav")
            self.send_header("X-Audio-Sample-Rate", str(SAMPLE_RATE))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(engine: Engine, host: str, port: int, default_voice: str) -> ThreadingHTTPServer:
    if host not in LOOPBACK:
        raise SystemExit(f"refusing to listen on {host!r}: this server has no authentication, use a loopback address")
    httpd = ThreadingHTTPServer((host, port), make_handler(engine, default_voice))
    httpd.daemon_threads = True
    return httpd


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--default-voice", default=kokoro_voices.DEFAULT_VOICE)
    parser.add_argument("--prime", action="append", default=[], metavar="VOICE",
                        help="also load this voice's language pipeline at start-up (repeatable)")
    parser.add_argument("--download", action="store_true", help="download the pinned model and exit (installer)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.download:
        check_offline_packages()
        print(download(args.model, args.revision))
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    engine = Engine(args.model, args.revision)
    httpd = serve(engine, args.host, args.port, args.default_voice)

    def load_and_prime() -> None:
        engine.load()
        # The language pipeline (G2P, and for English spaCy and torch) loads on first use and can
        # take tens of seconds; pay that at start-up for every configured voice, not on a reply.
        for voice in dict.fromkeys([args.default_voice, *args.prime]):
            if not (engine.loaded and kokoro_voices.is_valid_voice_id(voice)):
                continue
            try:
                engine.synthesize(".", voice, kokoro_voices.lang_code_for(voice), 1.0)
                log.info("pipeline primed for %s", voice)
            except Exception as exc:
                log.warning("priming %s failed: %s", voice, type(exc).__name__)

    threading.Thread(target=load_and_prime, name="kokoro-load", daemon=True).start()
    log.info("listening on http://%s:%d", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
