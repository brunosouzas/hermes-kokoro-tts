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
# exist in both, or that are ordinary words in the other language, are left out on purpose:
# "a", "do", "no", "as", "me", "so", "for" (pt "se for"), "era", "ate", "com" (".com").
_FUNCTION_WORDS: dict[str, frozenset[str]] = {
    "en": frozenset("""
        the and is are was were be been being to of in it its you your yours that this these those
        with on at by from have has had will would can could should not what which who whom
        when where why how there their they them we our us he she his her him i my an or but
        if then than about into over after before just also only very all any some each every
        does did done doing yes hi hello please thanks thank next today now here
    """.split()),
    "pt": frozenset("""
        o os um uma uns umas que não nao é e em para pra por pelo pela pelos pelas mais mas
        você voce vocês está esta estão são foi ser ter tem têm isso isto esse essa este
        aquele aquela como quando onde porque também tambem já ja ainda muito muita seu sua seus
        suas meu minha nosso nossa ele ela eles elas nós lhe dos das num numa ao aos à às
        sim olá ola obrigado obrigada hoje agora aqui depois antes então entao sobre até
        fazer feito pode podem vai vamos estou tenho
    """.split()),
}
# ã, õ and ç are Portuguese; the acute and circumflex accents also show up in English loanwords
# (café, résumé), so they are a weak signal: at most one point per word.
_PT_STRONG = re.compile(r"[ãõç]", re.IGNORECASE)
_PT_WEAK = re.compile(r"[âêôàáéíóú]", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
# Text a language signal must not come from: URLs, e-mail addresses, inline code, paths.
_NOT_PROSE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+|`[^`]*`|(?:\S*/\S+)")

# A sentence ends at . ! ? … (optionally followed by a closing quote or bracket) plus whitespace,
# or at a line break. A full stop after an abbreviation or an initial does not end a sentence.
_BOUNDARY = re.compile(r"([.!?…]+[\"'”’)\]]*)\s+|\n+")
_ABBREVIATIONS = frozenset("""
    dr dra drs sr sra srs srta mr mrs ms prof profa st jr sra vs etc eg ie ex av fig cap pág pag
    approx inc ltd dept min máx max nº vol ed
""".split())
_TOKEN_BEFORE = re.compile(r"(\S+)$")

SUPPORTED = tuple(_FUNCTION_WORDS)


def detect(text: str) -> Optional[str]:
    """``"pt"``, ``"en"`` or ``None`` when the sentence gives no clear signal."""
    prose = _NOT_PROSE.sub(" ", text)
    words = [w.lower() for w in _WORD.findall(prose)]
    if not words:
        return None
    scores = {lang: sum(1 for w in words if w in vocab) for lang, vocab in _FUNCTION_WORDS.items()}
    scores["pt"] += 2 * len(_PT_STRONG.findall(prose))
    scores["pt"] += sum(1 for w in words if _PT_WEAK.search(w) and not _PT_STRONG.search(w))
    ordered = sorted(scores.values(), reverse=True)
    if ordered[0] == 0 or ordered[0] == ordered[1]:
        return None
    return max(scores, key=lambda lang: scores[lang])


def _is_abbreviation(text_before: str) -> bool:
    match = _TOKEN_BEFORE.search(text_before)
    if not match:
        return False
    raw = match.group(1).lstrip("([\"'“‘").rstrip(".")
    token = raw.lower()
    # A single capital letter is an initial ("J. Smith"); short dotted forms ("e.g", "p.ex") too.
    initial = len(raw) == 1 and raw.isupper()
    return token in _ABBREVIATIONS or initial or ("." in token and len(token) <= 5 and token[0].isalpha())


_INLINE_CODE = re.compile(r"`[^`\n]*`")


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    # Never cut inside inline code: its punctuation is not prose. (URLs, paths and addresses
    # contain no whitespace, so a boundary, which needs whitespace, cannot fall inside them.)
    code = [m.span() for m in _INLINE_CODE.finditer(text)]
    for match in _BOUNDARY.finditer(text):
        if any(lo <= match.start() < hi for lo, hi in code):
            continue
        punct = match.group(1)
        if punct and punct.startswith(".") and len(punct.rstrip("\"'”’)]")) == 1 \
                and _is_abbreviation(text[start:match.start()]):
            continue
        end = match.end(1) if punct else match.start()
        parts.append(text[start:end])
        start = match.end()
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


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
