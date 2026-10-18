import pytest

import kokoro_voices as v


@pytest.mark.parametrize("voice", ["af_heart", "pf_dora", "bm_george", "zf_xiaoxiao", "em_santa"])
def test_valid_voice_ids(voice):
    assert v.is_valid_voice_id(voice)


@pytest.mark.parametrize("voice", ["", "heart", "../af_heart", "af_heart.safetensors", "/tmp/af_heart",
                                   "xf_unknownlang", "aX_caps", "af-heart", None, 3])
def test_invalid_voice_ids_are_rejected(voice):
    assert not v.is_valid_voice_id(voice)


def test_language_comes_from_the_voice():
    assert v.lang_code_for("pf_dora") == "p"
    assert v.lang_code_for("bf_emma") == "b"


def test_language_override_accepts_code_or_bcp47():
    assert v.lang_code_for("af_heart", "p") == "p"
    assert v.lang_code_for("af_heart", "pt-br") == "p"
    with pytest.raises(ValueError):
        v.lang_code_for("af_heart", "klingon")


def test_invalid_voice_raises_without_override():
    with pytest.raises(ValueError):
        v.lang_code_for("../etc/passwd")


def test_describe():
    assert v.describe("pf_dora") == {"id": "pf_dora", "display": "Dora (Brazilian Portuguese)",
                                     "language": "pt-br", "gender": "female"}
