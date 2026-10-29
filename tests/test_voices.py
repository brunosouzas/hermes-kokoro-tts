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


def test_voice_map_keeps_usable_entries_and_explains_the_rest():
    mapping, problems = v.voice_map({"PT": "pf_dora", "it": "im_nicola", "es": "ef_dora", "en": "../x"},
                                    ("en", "pt", "it"))
    assert mapping == {"pt": "pf_dora", "it": "im_nicola"}
    assert len(problems) == 2
    assert "voices.es" in problems[0] and "ef_dora" in problems[0] and "en, pt, it" in problems[0]
    assert "voices.en" in problems[1] and "not a Kokoro voice id and is ignored" in problems[1]


def test_voice_map_duplicate_key_is_explained():
    mapping, problems = v.voice_map({"PT": "pf_dora", "pt": "pm_alex"}, ("pt",))
    assert mapping == {"pt": "pm_alex"} and len(problems) == 1 and "twice" in problems[0]


def test_voice_map_that_is_not_a_map_is_explained():
    assert v.voice_map(None, ("en",)) == ({}, [])
    assert v.voice_map({}, ("en",)) == ({}, [])
    mapping, problems = v.voice_map("en=bf_emma", ("en",))
    assert mapping == {} and len(problems) == 1 and "map" in problems[0]
