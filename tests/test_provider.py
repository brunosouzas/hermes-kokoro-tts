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
    assert p.name == "kokoro" and p.streams_pcm is True and p.stream_sample_rate == 24000


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


def test_bad_settings_fall_back_to_defaults(fake_server):
    p = provider(fake_server, voice="../x", speed="fast")
    assert p.default_voice() == "af_heart" and p._speed(None) == 1.0
    assert p._speed(10) == 4.0
