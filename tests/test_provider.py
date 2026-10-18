"""The Hermes provider against a stand-in HTTP server."""

import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kokoro_tts.kokoro_provider import KokoroServerError, KokoroTTSProvider

PCM = b"\x01\x00" * 2400


class Recorder(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"voices": ["af_heart", "pf_dora", "bad/voice"]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Recorder.requests.append(req)
        if req["input"] == "fail":
            body = json.dumps({"error": {"message": "voice 'x' is not in the installed model"}}).encode()
            self.send_response(404)
        elif req["response_format"] == "pcm":
            body = PCM
            self.send_response(200)
        else:
            import io
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000), w.writeframes(PCM)
            body = buf.getvalue()
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def fake_server():
    Recorder.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()


def provider(port, **settings):
    values = {"port": port, **settings}
    return KokoroTTSProvider(lambda key, default=None: values.get(key, default))


def test_identity_and_streaming_contract(fake_server):
    p = provider(fake_server)
    assert p.name == "kokoro-mlx" and p.streams_pcm is True and p.stream_sample_rate == 24000


def test_provider_name_does_not_shadow_a_builtin():
    # Hermes ignores a plugin provider whose name is a built-in; "kokoro" itself is proposed as one.
    p = provider(8770)
    reserved = {"kokoro"}
    try:
        from agent.tts_registry import _BUILTIN_NAMES
        reserved |= set(_BUILTIN_NAMES)
    except Exception:
        pass
    assert p.name not in reserved


def test_is_available_means_installed_and_never_touches_the_network(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr("kokoro_tts.kokoro_provider.Path.home", classmethod(lambda cls: tmp_path))

    def no_network(*args, **kwargs):
        raise AssertionError("is_available() must not open a socket")
    monkeypatch.setattr(socket, "create_connection", no_network)

    p = provider(8770)
    assert p.is_available() is False
    agents = tmp_path / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    (agents / "dev.hermes.kokoro-tts.plist").write_text("<plist/>")
    assert p.is_available() is True


@pytest.mark.parametrize("value", [0, 70000, -1, "x", None])
def test_out_of_range_port_falls_back_without_raising(value):
    assert provider(value).port == 8770


def test_synthesize_writes_wav_and_fixes_extension(fake_server, tmp_path):
    p = provider(fake_server, voice="pf_dora", speed=1.2)
    written = p.synthesize("Olá, Bruno.", str(tmp_path / "reply.mp3"), format="mp3")
    assert written.endswith("reply.wav")
    with wave.open(written) as w:
        assert w.getframerate() == 24000 and w.getnframes() == 2400
    assert Recorder.requests[-1] == {"input": "Olá, Bruno.", "voice": "pf_dora", "speed": 1.2,
                                     "response_format": "wav", "lang_code": "p"}
    assert not list(tmp_path.glob("*.part"))


def test_explicit_voice_wins_over_setting(fake_server, tmp_path):
    p = provider(fake_server, voice="pf_dora")
    p.synthesize("hi", str(tmp_path / "a.wav"), voice="bf_emma")
    assert Recorder.requests[-1]["voice"] == "bf_emma" and Recorder.requests[-1]["lang_code"] == "b"


def test_language_override(fake_server, tmp_path):
    p = provider(fake_server, voice="af_heart", language="pt-br")
    p.synthesize("oi", str(tmp_path / "a.wav"))
    assert Recorder.requests[-1]["lang_code"] == "p"


def test_path_like_voice_is_refused_before_any_request(fake_server, tmp_path):
    p = provider(fake_server)
    with pytest.raises(KokoroServerError, match="not a Kokoro voice id"):
        p.synthesize("x", str(tmp_path / "a.wav"), voice="/Users/me/voices/pf_dora.safetensors")
    assert Recorder.requests == []


def test_server_error_message_is_surfaced(fake_server, tmp_path):
    with pytest.raises(KokoroServerError, match="404: voice 'x' is not in the installed model"):
        provider(fake_server).synthesize("fail", str(tmp_path / "a.wav"))


def test_unreachable_server_explains_next_step(tmp_path):
    with pytest.raises(KokoroServerError, match="hermes kokoro status"):
        provider(9).synthesize("x", str(tmp_path / "a.wav"))


def test_stream_yields_raw_pcm(fake_server):
    chunks = list(provider(fake_server).stream("hello", format="pcm"))
    assert b"".join(chunks) == PCM and Recorder.requests[-1]["response_format"] == "pcm"


def test_stream_other_formats_fall_back(fake_server):
    with pytest.raises(NotImplementedError):
        provider(fake_server).stream("hello", format="opus")


def test_list_voices_filters_and_describes(fake_server):
    assert [v["id"] for v in provider(fake_server).list_voices()] == ["af_heart", "pf_dora"]


def test_proxy_environment_is_ignored_for_loopback(fake_server, tmp_path, monkeypatch):
    # A proxy that would receive the text if urllib's environment proxies were honoured.
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(var, "http://proxy.invalid:3128")
    for var in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(var, raising=False)
    # urlopen() caches a global opener whose proxies were read earlier; drop it so the
    # environment above would apply to any code path that still uses the default opener.
    import urllib.request
    monkeypatch.setattr(urllib.request, "_opener", None)
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    p = provider(fake_server)
    p.synthesize("private words", str(tmp_path / "a.wav"))
    assert Recorder.requests[-1]["input"] == "private words"
    assert [v["id"] for v in p.list_voices()] == ["af_heart", "pf_dora"]


def test_provider_and_installer_agree_on_ports():
    from kokoro_tts import kokoro_cli
    for value in (0, 70000, "x", None, 8771, "8772"):
        settings = {"port": value}
        lookup = lambda k, d=None: settings.get(k, d)  # noqa: E731
        assert KokoroTTSProvider(lookup).port == kokoro_cli.service_config(lookup).port


BILINGUAL = {"voice": "pf_dora", "voices": {"pt": "pf_dora", "en": "bf_emma"}}
MIXED = "Olá, Bruno. O deploy terminou. The pipeline is green and the tests passed."


def test_synthesize_uses_one_voice_per_language_and_joins_the_audio(fake_server, tmp_path):
    written = provider(fake_server, **BILINGUAL).synthesize(MIXED, str(tmp_path / "a.wav"))
    sent = [(r["input"], r["voice"], r["lang_code"], r["response_format"]) for r in Recorder.requests]
    assert sent == [("Olá, Bruno. O deploy terminou.", "pf_dora", "p", "pcm"),
                    ("The pipeline is green and the tests passed.", "bf_emma", "b", "pcm")]
    with wave.open(written) as w:
        assert w.getframerate() == 24000 and w.getnframes() == 2 * 2400


def test_stream_uses_one_request_per_language_run(fake_server):
    chunks = list(provider(fake_server, **BILINGUAL).stream(MIXED, format="pcm"))
    assert b"".join(chunks) == PCM * 2
    assert [r["voice"] for r in Recorder.requests] == ["pf_dora", "bf_emma"]


def test_single_language_text_is_one_request(fake_server, tmp_path):
    provider(fake_server, **BILINGUAL).synthesize("The pipeline is green.", str(tmp_path / "a.wav"))
    assert [(r["voice"], r["response_format"]) for r in Recorder.requests] == [("bf_emma", "wav")]


def test_explicit_voice_turns_switching_off(fake_server, tmp_path):
    provider(fake_server, **BILINGUAL).synthesize(MIXED, str(tmp_path / "a.wav"), voice="pm_alex")
    assert [r["voice"] for r in Recorder.requests] == ["pm_alex"]


def test_language_override_is_ignored_in_per_language_mode(fake_server, tmp_path):
    provider(fake_server, language="e", **BILINGUAL).synthesize(MIXED, str(tmp_path / "a.wav"))
    assert [r["lang_code"] for r in Recorder.requests] == ["p", "b"]


def test_invalid_voice_map_entries_are_ignored(fake_server):
    p = provider(fake_server, voice="pf_dora", voices={"en": "../x", "PT": "pf_dora", "klingon": "af_heart"})
    assert p.language_voices() == {"pt": "pf_dora"}
    assert provider(fake_server, voices="en=bf_emma").language_voices() == {}


def test_warm_primes_each_configured_voice_once(fake_server):
    p = provider(fake_server, **BILINGUAL)
    p.warm()
    import time
    for _ in range(50):
        if len(Recorder.requests) >= 2:
            break
        time.sleep(0.05)
    assert [r["voice"] for r in Recorder.requests] == ["pf_dora", "bf_emma"]


def test_an_empty_later_run_fails_instead_of_writing_a_partial_file(fake_server, tmp_path, monkeypatch):
    real_post = KokoroTTSProvider._post
    calls = []

    def flaky(self, text, voice, speed, fmt, timeout, auto=False):
        calls.append(text)
        if len(calls) == 2:
            import io

            class Empty(io.BytesIO):
                pass
            return Empty(b"")
        return real_post(self, text, voice, speed, fmt, timeout, auto)

    monkeypatch.setattr(KokoroTTSProvider, "_post", flaky)
    out = tmp_path / "a.wav"
    with pytest.raises(KokoroServerError, match="part of the text"):
        provider(fake_server, **BILINGUAL).synthesize(MIXED, str(out))
    assert not out.exists() and not list(tmp_path.glob("*.part"))


def test_closing_the_stream_before_reading_releases_the_first_response(fake_server, monkeypatch):
    opened = []
    real_post = KokoroTTSProvider._post

    def tracking(self, *args, **kwargs):
        resp = real_post(self, *args, **kwargs)
        opened.append(resp)
        return resp

    monkeypatch.setattr(KokoroTTSProvider, "_post", tracking)
    stream = provider(fake_server, **BILINGUAL).stream(MIXED, format="pcm")
    stream.close()
    assert len(opened) == 1 and opened[0].closed


def test_closing_mid_stream_closes_the_active_response(fake_server, monkeypatch):
    opened = []
    real_post = KokoroTTSProvider._post

    def tracking(self, *args, **kwargs):
        resp = real_post(self, *args, **kwargs)
        opened.append(resp)
        return resp

    monkeypatch.setattr(KokoroTTSProvider, "_post", tracking)
    stream = provider(fake_server, **BILINGUAL).stream(MIXED, format="pcm")
    next(stream)
    stream.close()
    assert all(r.closed for r in opened) and len(opened) == 1


def test_warm_in_single_voice_mode_keeps_the_language_override(fake_server):
    import time
    provider(fake_server, voice="pf_dora", language="b").warm()
    for _ in range(50):
        if Recorder.requests:
            break
        time.sleep(0.05)
    assert [(r["voice"], r["lang_code"]) for r in Recorder.requests] == [("pf_dora", "b")]


def test_single_voice_mode_is_unchanged(fake_server, tmp_path):
    provider(fake_server, voice="pf_dora", language="b").synthesize(MIXED, str(tmp_path / "a.wav"))
    assert [(r["input"], r["voice"], r["lang_code"], r["response_format"]) for r in Recorder.requests] == [
        (MIXED, "pf_dora", "b", "wav")]


def test_english_words_go_with_non_english_runs_only(fake_server, tmp_path):
    p = provider(fake_server, english_words=["pipeline", " card ", "", 7, "pipeline"], english_accent="uk",
                 **BILINGUAL)
    p.synthesize(MIXED, str(tmp_path / "a.wav"))
    pt_req, en_req = Recorder.requests
    assert pt_req["english_words"] == ["pipeline", "card"] and pt_req["english_accent"] == "b"
    assert "english_words" not in en_req  # an English voice already says them in English


def test_no_english_words_sends_nothing_extra(fake_server, tmp_path):
    provider(fake_server, voice="pf_dora").synthesize("Oi.", str(tmp_path / "a.wav"))
    assert "english_words" not in Recorder.requests[-1]


def test_words_command_adds_removes_and_saves(tmp_path):
    from argparse import Namespace
    from kokoro_tts import kokoro_cli
    store = {"english_words": ["pipeline"]}
    handler = kokoro_cli.make_handler(lambda k, d=None: store.get(k, d), lambda: None,
                                      save=lambda k, v: store.__setitem__(k, v))
    assert handler(Namespace(kokoro_command="words", action="add", entries=["card", "Pipeline", "pull  request"])) == 0
    assert store["english_words"] == ["pipeline", "card", "pull request"]
    assert handler(Namespace(kokoro_command="words", action="remove", entries=["CARD"])) == 0
    assert store["english_words"] == ["pipeline", "pull request"]
    assert handler(Namespace(kokoro_command="words", action="add", entries=[])) == 2


def test_words_command_rejects_what_synthesis_would_ignore():
    from argparse import Namespace
    from kokoro_tts import kokoro_cli, kokoro_loanwords
    store = {}
    store["english_words"] = [f"word{chr(97 + i % 26)}{chr(97 + i // 26 % 26)}" for i in range(kokoro_loanwords.MAX_ENTRIES - 1)]
    handler = kokoro_cli.make_handler(lambda k, d=None: store.get(k, d), lambda: None,
                                      save=lambda k, v: store.__setitem__(k, v))
    code = handler(Namespace(kokoro_command="words", action="add",
                             entries=["4.9", "one two three four five", "x" * 65, "fits", "toomany"]))
    assert code == 1
    assert store["english_words"][-1] == "fits" and len(store["english_words"]) == kokoro_loanwords.MAX_ENTRIES
    assert "toomany" not in store["english_words"] and "4.9" not in store["english_words"]


def test_bad_settings_fall_back_to_defaults(fake_server):
    p = provider(fake_server, voice="../x", speed="fast")
    assert p.default_voice() == "af_heart" and p._speed(None) == 1.0
    assert p._speed(10) == 4.0
