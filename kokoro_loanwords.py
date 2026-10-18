"""English words inside non-English sentences, pronounced in English by the same voice.

Kokoro speaks from phonemes. For a sentence that contains words from the user's dictionary, the
whole sentence is phonemized with the voice's own language (so stress and vowel quality of the
surrounding words stay natural) and only the phoneme words of the listed words are swapped for
English phonemes. Pure standard library: the two phonemizers are passed in.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Optional

G2P = Callable[[str], str]

# Words as espeak counts them: letters (with inner hyphens or apostrophes) or digit runs.
_WORD = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*|\d+", re.UNICODE)
_TRAILING_PUNCT = re.compile(r"[.,!?;:…]+$")
MAX_ENTRY_WORDS = 4
MAX_ENTRIES = 500


def normalize(entries: Iterable[object]) -> list[tuple[str, ...]]:
    """Dictionary entries as lowercase word tuples; anything that is not a short phrase of
    letters is dropped. Longest entries first, so "pull request" wins over "pull"."""
    seen: dict[tuple[str, ...], None] = {}
    for entry in entries:
        if not isinstance(entry, str):
            continue
        words = tuple(w.lower() for w in _WORD.findall(entry))
        if not words or len(words) > MAX_ENTRY_WORDS or any(w.isdigit() for w in words):
            continue
        seen.setdefault(words, None)
        if len(seen) >= MAX_ENTRIES:
            break
    return sorted(seen, key=len, reverse=True)


def _matches(words: list[str], start: int, entry: tuple[str, ...]) -> bool:
    if start + len(entry) > len(words):
        return False
    for offset, expected in enumerate(entry):
        word = words[start + offset].lower()
        last = offset == len(entry) - 1
        # A plural of the last word counts too: "cards" for "card", "pull requests".
        if word != expected and not (last and word == expected + "s"):
            return False
    return True


def find(text: str, entries: list[tuple[str, ...]]) -> list[tuple[int, int]]:
    """``(first_word, end_word)`` index spans of dictionary hits, non-overlapping."""
    words = _WORD.findall(text)
    spans, index = [], 0
    while index < len(words):
        for entry in entries:
            if _matches(words, index, entry):
                spans.append((index, index + len(entry)))
                index += len(entry)
                break
        else:
            index += 1
    return spans


def splice(text: str, entries: list[tuple[str, ...]], native: G2P, english: G2P) -> Optional[str]:
    """Phonemes for ``text`` with the listed words in English, or ``None`` when no listed word
    occurs or the phoneme words cannot be lined up with the text's words (the caller then
    synthesizes the text normally)."""
    spans = find(text, entries)
    if not spans:
        return None
    words = _WORD.findall(text)
    tokens = native(text).strip().split()
    if len(tokens) != len(words):
        return None
    out: list[str] = []
    index = 0
    for start, end in spans:
        out.extend(tokens[index:start])
        phrase = " ".join(words[start:end])
        spoken = english(phrase).strip()
        if not spoken or "❓" in spoken:  # the English lexicon does not know it: keep native
            out.extend(tokens[start:end])
        else:
            trail = _TRAILING_PUNCT.search(tokens[end - 1])
            out.append(spoken + (trail.group(0) if trail else ""))
        index = end
    out.extend(tokens[index:])
    return " ".join(out)
