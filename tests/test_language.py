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
    # Review regressions: shared words, loanword accents, URLs and code must not decide.
    ("Me avisa quando terminar.", "pt"),
    ("Me avisa.", None),
    ("Please update my résumé.", "en"),
    ("The café is open.", "en"),
    ("Se for possível, me liga.", "pt"),
    ("Leia https://example.com/in/it/the/docs agora.", "pt"),
    ("Rode `git push origin main` depois do merge.", "pt"),
    ("Run `npm install o pacote` and then restart it.", "en"),
    ("The era of local models is here.", "en"),
])
def test_detect(text, expected):
    assert lang.detect(text) == expected


def test_split_sentences():
    assert lang.split_sentences("One. Two!  Three?\nFour") == ["One.", "Two!", "Three?", "Four"]


@pytest.mark.parametrize("text,expected", [
    ("O Dr. Brown me ligou hoje. Depois eu volto.", ["O Dr. Brown me ligou hoje.", "Depois eu volto."]),
    ("Mr. Smith is here. He is waiting.", ["Mr. Smith is here.", "He is waiting."]),
    ("Use e.g. the default. Then restart.", ["Use e.g. the default.", "Then restart."]),
    ("Talk to J. Smith today. Thanks.", ["Talk to J. Smith today.", "Thanks."]),
    ("A versão 4.9 saiu. Atualize.", ["A versão 4.9 saiu.", "Atualize."]),
    ("Veja example.com. Depois volte.", ["Veja example.com.", "Depois volte."]),
    ("Ele disse \"pronto.\" Then he left.", ["Ele disse \"pronto.\"", "Then he left."]),
    ("- primeiro item\n- second item", ["- primeiro item", "- second item"]),
    ("Fim de frase com é. Depois outra.", ["Fim de frase com é.", "Depois outra."]),
])
def test_split_sentences_edge_cases(text, expected):
    assert lang.split_sentences(text) == expected


def test_abbreviation_does_not_switch_voice_mid_sentence():
    assert lang.plan("O Dr. Brown me ligou hoje.", VOICES, "pf_dora") == [("O Dr. Brown me ligou hoje.", "pf_dora")]


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
