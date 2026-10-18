"""English words inside non-English sentences.

Most tests replay real phonemizer output recorded from the pinned Kokoro pipelines
(``fixtures/g2p.json``, written by ``record_g2p.py``): espeak pt-br for the voice's language and
Kokoro's English lexicon (US ``a``, UK ``b``). A text the fixture does not hold raises KeyError,
so a test can never pass on an unrecorded call.
"""

import json
from pathlib import Path

import pytest

import kokoro_loanwords as lw

G2P = json.loads((Path(__file__).parent / "fixtures" / "g2p.json").read_text(encoding="utf-8"))


def nfc(text):
    import unicodedata
    return unicodedata.normalize("NFC", text) if text is not None else None


def native(text):
    return G2P["p"][text]


def uk(text):
    return G2P["b"][text]


def us(text):
    return G2P["a"][text]


ENTRIES = lw.normalize(["pipeline", "card", "deploy", "board", "merge", "pull request", "feature", "e-mail"])


@pytest.mark.parametrize("text,expected", [
    ("O pipeline terminou e o card do deploy já está no board.",
     "ʊ pˈIplIn tˌeɾəminˈow i ʊ kˈɑːd dʊ dɪplˈY ʒˈa estˌa nʊ bˈɔːd."),
    ("Faz o merge do pull request quando puder.",
     "faz ʊ mˈɜːʤ dʊ pˈʊl ɹɪkwˈɛst kwˈɐ̃ŋdʊ pudˈɛr."),
    # plurals; "cards" is /kaɾəds/ alone but /kaɾədz/ here, and still lines up
    ("Revisei os cards e as features.", "xˌevizˈA ʊs kˈɑːdz i as fˈiːʧəz."),
    # punctuation espeak keeps on the word survives the swap
    ("Abre o (card).", "ˈabri ˈɔ (kˈɑːd)."),
    ("Ele disse \"card\" duas vezes.", "ˌely ʤˈisy \"kˈɑːd\" dˈuæz vˈezys."),
    ("Deploy!", "dɪplˈY!"),
    # hyphens, apostrophes and numbers do not break the alignment
    ("Manda um e-mail sobre o deploy.", "mˈɐ̃ŋdæ ũŋ ˈiːmˌAl sˈobri ʊ dɪplˈY."),
    ("Um copo d'água antes do deploy.", "ũŋ kˈɔpʊ dˈaɡwæ ˈɐ̃ŋʧyz dʊ dɪplˈY."),
    ("A versão 4.9 do pipeline saiu.", "a veɾəsˈɐ̃ʊ̃ kwˈatrʊ.nˈɔvy dʊ pˈIplIn saˈiʊ."),
    ("Temos 3 cards no board.", "tˌemʊs trˈes kˈɑːdz nʊ bˈɔːd."),
])
def test_real_phonemes_uk(text, expected):
    # espeak emits decomposed nasal vowels (u + U+0303); compare in one normal form.
    assert nfc(lw.splice(text, ENTRIES, native, uk)) == nfc(expected)


def test_us_accent():
    assert lw.splice("Abre o (card).", ENTRIES, native, us) == "ˈabri ˈɔ (kˈɑɹd)."


def test_phrase_never_matches_across_punctuation():
    assert lw.splice("O pull, request ficou para depois.", lw.normalize(["pull request"]), native, uk) is None


def test_whole_words_only():
    assert lw.splice("O cardápio chegou.", ENTRIES, native, uk) is None


def test_no_listed_word_means_normal_synthesis():
    assert lw.splice("O cardápio chegou.", lw.normalize(["board"]), native, uk) is None


def test_equal_count_misalignment_is_caught():
    # Same number of phoneme words, but shifted: "card"'s slot holds another word's phonemes.
    def shifted(text):
        if text == "Abre o (card).":
            return "ˈabri (kˈaɾəd). ˈɔ"
        return native(text)
    assert lw.splice("Abre o (card).", ENTRIES, shifted, uk) is None


def test_count_mismatch_falls_back():
    def merged(text):
        return "x " + native(text)
    assert lw.splice("Deploy!", ENTRIES, merged, uk) is None


@pytest.mark.parametrize("spoken", ["", "❓", "pˈʊl"])  # empty, unknown, wrong word count
def test_unusable_english_keeps_the_native_word(spoken):
    out = lw.splice("Faz o merge do pull request quando puder.", lw.normalize(["pull request"]),
                    native, lambda text: spoken)
    assert out == native("Faz o merge do pull request quando puder.")


def test_normalize_orders_phrases_first_and_drops_junk():
    entries = lw.normalize(["card", "pull request", "CARD", "", 3, "a b c d e", "v2", "4.9", "card"])
    assert entries == [("pull", "request"), ("card",)]


def test_entry_words():
    assert lw.entry_words("Pull  Request") == ("pull", "request")
    assert lw.entry_words("4.9") is None and lw.entry_words(7) is None and lw.entry_words("a b c d e") is None


@pytest.mark.parametrize("entries", [[], lw.normalize([])])
def test_empty_dictionary(entries):
    assert lw.splice("Deploy!", entries, native, uk) is None
