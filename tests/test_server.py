"""The HTTP server, with a fake model in place of MLX (runs on any platform)."""

import json
import threading
import urllib.error
import urllib.request
import wave
import io

import numpy as np
import pytest

import kokoro_server as ks


class FakeResult:
    def __init__(self, audio):
        self.audio = audio


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, text, voice, lang_code, speed):
        self.calls.append({"text": text, "voice": voice, "lang_code": lang_code, "speed": speed})
        yield FakeResult(np.full(2400, 0.5, dtype=np.float32))
        yield FakeResult(np.full(2400, -2.0, dtype=np.float32))  # clipped to -1.0


@pytest.fixture
def snapshot(tmp_path):
    (tmp_path / "voices").mkdir()
    for name in ("af_heart", "pf_dora"):
        (tmp_path / "voices" / f"{name}.safetensors").write_bytes(b"x")
    (tmp_path / "voices" / "not a voice.safetensors").write_bytes(b"x")
    return tmp_path


@pytest.fixture
def server(snapshot):
    model = FakeModel()
    engine = ks.Engine("repo/x", "a" * 40, loader=lambda: model, snapshot=snapshot)
    engine.load()
    httpd = ks.serve(engine, "127.0.0.1", 0, "af_heart")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", model, engine
    httpd.shutdown()


def post(url, payload, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(url + "/v1/audio/speech", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5)


def error_of(url, payload, raw=None):
    with pytest.raises(urllib.error.HTTPError) as info:
        post(url, payload, raw)
    return info.value.code, json.load(info.value)["error"]["message"]


def test_health_and_voices(server):
    url, _model, _engine = server
    health = json.load(urllib.request.urlopen(url + "/health"))
    assert health["ok"] and health["sample_rate"] == 24000 and health["revision"] == "a" * 40
    assert json.load(urllib.request.urlopen(url + "/v1/voices"))["voices"] == ["af_heart", "pf_dora"]


def test_wav_response_and_voice_resolved_to_snapshot_file(server, snapshot):
    url, model, _ = server
    with post(url, {"input": "Olá", "voice": "pf_dora", "response_format": "wav"}) as resp:
        assert resp.headers["Content-Type"] == "audio/wav"
        body = resp.read()
    with wave.open(io.BytesIO(body)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 24000, 4800)
        frames = np.frombuffer(w.readframes(4800), dtype="<i2")
    assert frames[0] == 16383 and frames[-1] == -32767
    assert model.calls[-1] == {"text": "Olá", "voice": str(snapshot / "voices" / "pf_dora.safetensors"),
                               "lang_code": "p", "speed": 1.0}


def test_pcm_response(server):
    url, _model, _ = server
    with post(url, {"input": "hi", "response_format": "pcm", "speed": 1.5}) as resp:
        assert resp.headers["X-Audio-Sample-Rate"] == "24000"
        assert len(resp.read()) == 4800 * 2


def test_lang_override(server):
    url, model, _ = server
    post(url, {"input": "hola", "voice": "af_heart", "lang_code": "e"}).read()
    assert model.calls[-1]["lang_code"] == "e"


@pytest.mark.parametrize("payload,status", [
    ({"input": ""}, 400),
    ({"input": "x", "voice": "../../etc/passwd"}, 400),
    ({"input": "x", "voice": "bf_missing"}, 404),
    ({"input": "x", "response_format": "mp3"}, 400),
    ({"input": "x", "speed": 9}, 400),
    ({"input": "x" * (ks.MAX_INPUT_CHARS + 1)}, 413),
])
def test_bad_requests(server, payload, status):
    url, model, _ = server
    code, _message = error_of(url, payload)
    assert code == status
    assert model.calls == []


def test_invalid_json(server):
    url, _model, _ = server
    assert error_of(url, None, raw=b"{nope")[0] == 400


def test_not_loaded_returns_503(snapshot):
    engine = ks.Engine("repo/x", "a" * 40, loader=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                       snapshot=snapshot)
    engine.load()
    httpd = ks.serve(engine, "127.0.0.1", 0, "af_heart")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert json.load(urllib.request.urlopen(url + "/health"))["ok"] is False
        assert error_of(url, {"input": "x"})[0] == 503
    finally:
        httpd.shutdown()


def test_refuses_non_loopback(snapshot):
    engine = ks.Engine("repo/x", "a" * 40, loader=lambda: None, snapshot=snapshot)
    with pytest.raises(SystemExit):
        ks.serve(engine, "0.0.0.0", 0, "af_heart")
