"""Per-sentence language detection and voice planning (standard library only).

Kokoro reads text with the pronunciation rules of the voice's language, so English read by a
Portuguese voice sounds wrong. With a language-to-voice map, each sentence goes to the voice of
its own language. Detection is a small function-word count: cheap, dependency-free and good
enough to tell whole sentences apart; it does not try to tag single foreign words.
"""

from __future__ import annotations

import re
from typing import Optional

# Function words that are common in one language and absent (or rare) in the other. Words that
# exist in both ("a", "do", "no", "as", "me", "so", "for" vs "por"...) are left out on purpose.
_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset("""
        the and is are was were be been being to of in it its you your yours that this these those
        with for on at by from have has had will would can could should not what which who whom
        when where why how there their they them we our us he she his her him i my me an or but
        if then than about into over after before just also only very all any some each every
        does did done doing yes hi hello please thanks thank next today now here
    """.split()),
    "pt": frozenset("""
        o os um uma uns umas que não nao é e em para pra com por pelo pela pelos pelas mais mas
        você voce vocês está esta estão são foi era ser ter tem têm isso isto esse essa este
        aquele aquela como quando onde porque também tambem já ja ainda muito muita seu sua seus
        suas meu minha nosso nossa ele ela eles elas nós nos lhe dos das num numa ao aos à às
        sim olá ola obrigado obrigada hoje agora aqui depois antes então entao sobre até ate
        fazer feito pode podem vai vamos estou tenho
    """.split()),
}
# Letters that only Portuguese (of the two) uses.
_PT_LETTERS = re.compile(r"[ãõçâêôàáéíóú]", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")

SUPPORTED = tuple(_FUNCTION_WORDS)


def detect(text: str) -> Optional[str]:
    """``"pt"``, ``"en"`` or ``None`` when the sentence gives no clear signal."""
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return None
    scores = {lang: sum(1 for w in words if w in vocab) for lang, vocab in _FUNCTION_WORDS.items()}
    scores["pt"] += 2 * len(_PT_LETTERS.findall(text))
    best = max(scores, key=lambda lang: scores[lang])
    ordered = sorted(scores.values(), reverse=True)
    if ordered[0] == 0 or ordered[0] == ordered[1]:
        return None
    return best


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text) if part and part.strip()]


def plan(text: str, voices: dict[str, str], default_voice: str) -> list[tuple[str, str]]:
    """Split ``text`` into consecutive ``(text, voice)`` runs.

    A sentence with no clear language keeps the previous sentence's voice (the first one falls
    back to ``default_voice``); neighbouring sentences with the same voice are merged so each run
    is one request.
    """
    runs: list[tuple[str, str]] = []
    current = default_voice
    for sentence in split_sentences(text):
        lang = detect(sentence)
        if lang is not None:
            current = voices.get(lang, default_voice)
        if runs and runs[-1][1] == current:
            runs[-1] = (f"{runs[-1][0]} {sentence}", current)
        else:
            runs.append((sentence, current))
    return runs or [(text, default_voice)]
