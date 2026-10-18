"""English words inside non-English sentences: matching and phoneme splicing with fake phonemizers."""

import pytest

import kokoro_loanwords as lw


def native(text):
    # One "phoneme word" per text word, punctuation kept on the word, like espeak.
    import re
    return " ".join("N:" + w for w in re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*[.,!?;:…]*|\d+[.,!?;:…]*", text))


def english(text):
    return " ".join("E:" + w.lower() for w in text.split())


ENTRIES = lw.normalize(["pipeline", "card", "Pull Request", "deploy"])


def test_normalize_orders_phrases_first_and_drops_junk():
    entries = lw.normalize(["card", "pull request", "CARD", "", 3, "a b c d e", "v2", "card"])
    assert entries == [("pull", "request"), ("card",)]


def test_splice_swaps_only_listed_words_and_keeps_punctuation():
    text = "O pipeline terminou e o card do deploy já está pronto."
    assert lw.splice(text, ENTRIES, native, english) == (
        "N:O E:pipeline N:terminou N:e N:o E:card N:do E:deploy N:já N:está N:pronto.")


def test_trailing_punctuation_of_a_listed_word_is_kept():
    assert lw.splice("Abre o card.", ENTRIES, native, english) == "N:Abre N:o E:card."


def test_phrases_and_plurals():
    out = lw.splice("Revisa os pull requests e os cards.", ENTRIES, native, english)
    assert out == "N:Revisa N:os E:pull E:requests N:e N:os E:cards."


def test_case_insensitive():
    assert "E:pipeline" in lw.splice("PIPELINE quebrou", ENTRIES, native, english)


def test_whole_words_only():
    assert lw.splice("O cardápio chegou.", ENTRIES, native, english) is None


def test_no_listed_word_means_normal_synthesis():
    assert lw.splice("Nada para trocar aqui.", ENTRIES, native, english) is None


def test_misaligned_phonemes_fall_back():
    def merged(text):  # a phonemizer that joins two words breaks the alignment
        return "N:one-token " + native(text)
    assert lw.splice("O card chegou.", ENTRIES, merged, english) is None


def test_unknown_english_word_keeps_the_native_pronunciation():
    def unknown(text):
        return "❓"
    assert lw.splice("O card chegou.", ENTRIES, native, unknown) == "N:O N:card N:chegou."


@pytest.mark.parametrize("entries", [[], lw.normalize([])])
def test_empty_dictionary(entries):
    assert lw.splice("O card chegou.", entries, native, english) is None
