import pytest

import kokoro_language as lang

VOICES = {"pt": "pf_dora", "en": "bf_emma"}


@pytest.mark.parametrize("text,expected", [
    ("Hi Bruno. Your deployment finished and the pipeline is green.", "en"),
    ("What is next on the board today?", "en"),
    ("Olá, Bruno. O deploy terminou e o pipeline está verde.", "pt"),
    ("Hoje não tem nada marcado na sua agenda.", "pt"),
    ("Você pode revisar isso depois?", "pt"),
    ("Ação concluída", "pt"),  # accents alone are a Portuguese signal
    ("MuleSoft Anypoint", None),  # names only: no signal
    ("12:30", None),
    ("", None),
])
def test_detect(text, expected):
    assert lang.detect(text) == expected


def test_split_sentences():
    assert lang.split_sentences("One. Two!  Three?\nFour") == ["One.", "Two!", "Three?", "Four"]


def test_plan_switches_voice_per_sentence_and_merges_neighbours():
    text = "Olá, Bruno. O deploy terminou. The pipeline is green and the tests passed. Até amanhã!"
    assert lang.plan(text, VOICES, "pf_dora") == [
        ("Olá, Bruno. O deploy terminou.", "pf_dora"),
        ("The pipeline is green and the tests passed.", "bf_emma"),
        ("Até amanhã!", "pf_dora"),
    ]


def test_unclear_sentence_keeps_previous_voice():
    text = "The build is done. MuleSoft 4.9. It is deployed."
    assert lang.plan(text, VOICES, "pf_dora") == [("The build is done. MuleSoft 4.9. It is deployed.", "bf_emma")]


def test_first_unclear_sentence_uses_default_voice():
    assert lang.plan("Anypoint.", VOICES, "pf_dora") == [("Anypoint.", "pf_dora")]


def test_unmapped_language_uses_default_voice():
    assert lang.plan("The pipeline is green.", {"pt": "pf_dora"}, "pf_dora") == [("The pipeline is green.", "pf_dora")]


def test_whitespace_only_text_is_returned_as_is():
    assert lang.plan("   ", VOICES, "pf_dora") == [("   ", "pf_dora")]
