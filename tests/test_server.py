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


@pytest.mark.parametrize("payload", [
    {"input": "x", "response_format": 123},
    {"input": "x", "lang_code": 5},
    {"input": "x", "voice": ["af_heart"]},
])
def test_wrong_field_types_are_400_not_crashes(server, payload):
    url, model, _ = server
    assert error_of(url, payload)[0] == 400
    assert model.calls == []


def test_failed_synthesis_never_logs_the_text(snapshot, caplog):
    secret = "my bank PIN is 4321"

    class Exploding:
        def generate(self, text, **kwargs):
            raise RuntimeError(f"cannot phonemize {text!r}")
            yield  # pragma: no cover

    engine = ks.Engine("repo/x", "a" * 40, loader=Exploding, snapshot=snapshot)
    engine.load()
    httpd = ks.serve(engine, "127.0.0.1", 0, "af_heart")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with caplog.at_level("DEBUG"):
            code, message = error_of(url, {"input": secret})
    finally:
        httpd.shutdown()
    # A model error is a 500, not the "not loaded" 503, and its message is never echoed.
    assert code == 500
    assert secret not in caplog.text and secret not in message


@pytest.mark.parametrize("exc_type", [KeyError, ValueError, LookupError, IndexError])
def test_generic_failure_logs_type_only(snapshot, caplog, exc_type):
    secret = "private sentence"

    class Broken:
        def generate(self, text, **kwargs):
            raise exc_type(text)
            yield  # pragma: no cover

    engine = ks.Engine("repo/x", "a" * 40, loader=Broken, snapshot=snapshot)
    engine.load()
    httpd = ks.serve(engine, "127.0.0.1", 0, "af_heart")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with caplog.at_level("DEBUG"):
            code, message = error_of(url, {"input": secret})
    finally:
        httpd.shutdown()
    assert code == 500 and exc_type.__name__ in caplog.text
    assert secret not in caplog.text and secret not in message


def test_offline_package_check(monkeypatch):
    monkeypatch.setattr(ks, "OFFLINE_PACKAGES", ("definitely_not_installed_pkg_xyz",))
    with pytest.raises(SystemExit, match="definitely_not_installed_pkg_xyz"):
        ks.check_offline_packages()
    monkeypatch.setattr(ks, "OFFLINE_PACKAGES", ("json",))
    ks.check_offline_packages()


@pytest.mark.parametrize("payload", [
    {"input": "x", "english_words": "pipeline"},
    {"input": "x", "english_words": [1, 2]},
    {"input": "x", "english_words": ["x" * 65]},
    {"input": "x", "english_words": ["w"] * (ks.MAX_ENGLISH_WORDS + 1)},
    {"input": "x", "english_accent": "au"},
])
def test_english_words_are_validated(server, payload):
    url, model, _ = server
    assert error_of(url, payload)[0] == 400 and model.calls == []


class FakePipeline:
    """Tags every phoneme word with the pipeline's language, so a test can see which language
    produced which word."""

    def __init__(self, calls, lang):
        self.calls = calls
        self.lang = lang

    def g2p(self, text):
        import re
        return " ".join(f"{self.lang}:" + w.lower() for w in re.findall(r"[^\W\d_]+[.,!?]*|\d+[.,!?]*", text)), None

    def generate_from_tokens(self, phonemes, voice, speed, model):
        self.calls.append({"phonemes": phonemes, "voice": voice})

        class Out:
            audio = np.full(2400, 0.25, dtype=np.float32)

        class Result:
            output = Out()
        yield Result()


class LoanwordModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.token_calls = []

    def _get_pipeline(self, lang):
        return FakePipeline(self.token_calls, lang)


@pytest.fixture
def loan_server(snapshot):
    model = LoanwordModel()
    engine = ks.Engine("repo/x", "a" * 40, loader=lambda: model, snapshot=snapshot)
    engine.load()
    httpd = ks.serve(engine, "127.0.0.1", 0, "af_heart")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", model
    httpd.shutdown()


def test_listed_words_are_spliced_per_sentence_and_the_rest_is_normal(loan_server, snapshot):
    url, model = loan_server
    text = "O pipeline terminou. Nada mais hoje."
    with post(url, {"input": text, "voice": "pf_dora", "response_format": "pcm",
                    "english_words": ["pipeline"], "english_accent": "b"}) as resp:
        body = resp.read()
    # Portuguese phonemes around, British English for the listed word, same Dora voice.
    assert model.token_calls == [{"phonemes": "p:o b:pipeline p:terminou.",
                                  "voice": str(snapshot / "voices" / "pf_dora.safetensors")}]
    assert [c["text"] for c in model.calls] == ["Nada mais hoje."]  # no listed word: normal path
    assert len(body) == (2400 + 4800) * 2


def test_us_accent_is_used_when_asked(loan_server):
    url, model = loan_server
    post(url, {"input": "O pipeline.", "voice": "pf_dora", "english_words": ["pipeline"],
               "english_accent": "a"}).read()
    assert model.token_calls[-1]["phonemes"] == "p:o a:pipeline."


def test_spliced_sentence_over_the_phoneme_limit_uses_normal_synthesis(loan_server):
    url, model = loan_server
    filler = " ".join(["palavra"] * 60)  # 60 x "p:palavra " is well over 510 phoneme characters
    sentence = f"O pipeline {filler}."
    post(url, {"input": sentence, "voice": "pf_dora", "english_words": ["pipeline"]}).read()
    assert model.token_calls == [] and model.calls[-1]["text"] == sentence


@pytest.mark.parametrize("length,spliced_path", [(ks.MAX_PHONEMES, True), (ks.MAX_PHONEMES + 1, False)])
def test_phoneme_limit_boundary(loan_server, monkeypatch, length, spliced_path):
    url, model = loan_server
    monkeypatch.setattr(ks.kokoro_loanwords, "splice", lambda *a, **k: "x" * length)
    post(url, {"input": "O pipeline.", "voice": "pf_dora", "english_words": ["pipeline"]}).read()
    assert bool(model.token_calls) is spliced_path
    assert (model.calls == []) is spliced_path


def test_english_voice_ignores_the_dictionary(loan_server):
    url, model = loan_server
    post(url, {"input": "The pipeline is green.", "voice": "af_heart", "english_words": ["pipeline"]}).read()
    assert model.token_calls == [] and model.calls[-1]["text"] == "The pipeline is green."


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
