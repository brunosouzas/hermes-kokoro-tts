"""Per-sentence language detection and voice planning (standard library only).

Kokoro reads text with the pronunciation rules of the voice's language, so English read by a
Portuguese voice sounds wrong. With a language-to-voice map, each sentence goes to the voice of
its own language. Detection is a small function-word count: cheap, dependency-free and good
enough to tell whole sentences apart; it does not try to tag single foreign words.
"""

from __future__ import annotations

import re
from typing import Optional

# Function words that are common in one language and absent (or rare) in the others. Words that
# exist in two of them, or that are ordinary words in another one, are left out on purpose:
# "a", "do", "no", "as", "me", "so", "for" (pt "se for"), "era", "ate", "com" (".com"); for Italian
# also "i", "in", "per", "con", "come", "ha" (English), "e", "da", "sei", "quando", "se", "tu",
# "lei", "tua", "posso", "deve", "sempre" (Portuguese), "lo"/"la" (the Portuguese "fazê-lo"),
# "sono", "sul", "prego", "né" (Portuguese sleep, south, nail, "não é"), "dove", "fare" (English), and "io",
# "mi", "sto" (socket.io, MI200, STO). An all-capitals token (DI, DEL, MA) is an acronym, not an
# Italian word.
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
        àquele àquela àqueles àquelas àquilo
        sim olá ola obrigado obrigada hoje agora aqui depois antes então entao sobre até
        fazer feito pode podem vai vamos estou tenho
    """.split()),
    "it": frozenset("""
        il le un uno una gli di del dello della delle degli nel nello nella nelle negli alla allo
        alle agli dal dalla dalle sulla sulle che non è c'è perché poiché benché finché affinché
        nonché anche ancora più molto molta molti poco questo questa questi queste quello quella
        siamo siete ho hai abbiamo avete hanno stato stata essere fatto ciao grazie ecco bene
        buongiorno buonasera allora però così già
        sì oggi domani ieri adesso qui dopo forse niente nulla tutto tutti tutta nessun nessuno
        nessuna cosa vorrei voglio puoi può stai lui noi voi loro mio mia tuo nostro nostra ma
    """.split()),
}
# ã, õ and ç are Portuguese; the acute and circumflex accents also show up in English loanwords
# (café, résumé), so they are a weak signal: at most one point per word. When Italian is scored
# too, an Italian function word (perché) gets no such point, and neither does the grave à:
# Portuguese only writes it in the crase (a function word above), and Italian writes it too
# (città, già).
_PT_STRONG = re.compile(r"[ãõç]", re.IGNORECASE)
_PT_WEAK = re.compile(r"[âêôàáéíóú]", re.IGNORECASE)
_PT_WEAK_NEXT_TO_IT = re.compile(r"[âêôáéíóú]", re.IGNORECASE)
# è, ì, ò and ù are Italian (Portuguese has no such letters): one point per word.
_IT_ACCENT = re.compile(r"[èìòù]", re.IGNORECASE)
# Italian elision: "l'aggiornamento", "un'altra", "dell'anno", with either apostrophe. At least
# two letters after it, so English "all's" does not count.
_IT_ELISION = re.compile(r"(?<![^\W\d_])(?:l|un|dell|nell|dall|sull|all|quell|quest|nessun)['’][^\W\d_]{2,}",
                         re.IGNORECASE)
# Italian writes no function word inside a hyphenated compound or an identifier, so
# "non-blocking", "non-2xx", "un-hide", "use_non_blocking" and the Portuguese "fazê-lo" give
# Italian no point.
_COMPOUND = re.compile(r"\S*[-_]\S*")
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
# Detected whatever the voice map says, as before Italian existed. Any other language is scored
# only when it has a voice, so its false positives cannot move a sentence for someone who never
# asked for it (an unmapped language would switch the sentence to the default voice).
ALWAYS_DETECTED = ("en", "pt")


def languages_for(voices: dict[str, str]) -> tuple[str, ...]:
    return tuple(lang for lang in SUPPORTED if lang in ALWAYS_DETECTED or lang in voices)


def detect(text: str, languages: tuple[str, ...] = SUPPORTED) -> Optional[str]:
    """The language of ``languages`` (``"pt"``, ``"en"``, ``"it"``) the sentence is in, or
    ``None`` when it gives no clear signal."""
    prose = _NOT_PROSE.sub(" ", text)
    words = [w.lower() for w in _WORD.findall(prose)]
    if not words:
        return None
    italian = "it" in languages
    scores = {lang: sum(1 for w in words if w in _FUNCTION_WORDS[lang]) for lang in languages if lang != "it"}
    if "pt" in scores:
        weak = _PT_WEAK_NEXT_TO_IT if italian else _PT_WEAK
        scores["pt"] += 2 * len(_PT_STRONG.findall(prose))
        scores["pt"] += sum(1 for w in words if weak.search(w) and not _PT_STRONG.search(w)
                            and not (italian and w in _FUNCTION_WORDS["it"]))
    if italian:
        scores["it"] = sum(1 for w in _WORD.findall(_COMPOUND.sub(" ", prose))
                           if w.lower() in _FUNCTION_WORDS["it"] and not (len(w) > 1 and w.isupper()))
        scores["it"] += sum(1 for w in words if _IT_ACCENT.search(w)) + len(_IT_ELISION.findall(prose))
    ordered = sorted(scores.values(), reverse=True) + [0]
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
    languages = languages_for(voices)
    for sentence in split_sentences(text):
        lang = detect(sentence, languages)
        if lang is not None:
            current = voices.get(lang, default_voice)
        if runs and runs[-1][1] == current:
            runs[-1] = (f"{runs[-1][0]} {sentence}", current)
        else:
            runs.append((sentence, current))
    return runs or [(text, default_voice)]
